from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed, manual_attack_contact, delete_unit, add_unit, set_unit_to_mission, auto_attack_contact
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import Multi_Side_FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
from scripts.single_agent_1v1.Mylib import *
import numpy as np
from collections import deque
import time
import random
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize, cosine_similarity
from copy import deepcopy   
import torch.optim as optim

import logging

# 導入深度強化學習模型
from module.batch_agent.FeUdal_agent import (
    Feudal_ManagerAgent, Feudal_WorkerAgent
)

def compute_gae(deltas: torch.Tensor, gamma: float, lam: float, done_mask: torch.Tensor):
    """
    deltas: [T, …]    已計算好的 δ_t
    done_mask: [T, …]  0=not done, 1=done  (用來重置尾端)
    回傳 same shape 的 GAE advantages
    """
    T = deltas.size(0)
    adv = torch.zeros_like(deltas)
    running = torch.zeros_like(deltas[0])
    for t in reversed(range(T)):
        running = deltas[t] + gamma * lam * (1 - done_mask[t]) * running
        adv[t] = running
    return adv

class MyAgent(BaseAgent):
    class EnemyInfo:
        """
        敵方單位的信息追蹤類
        用於跟踪敵方單位的存活狀態、數量變化等資訊
        """
        def __init__(self, player_side, enemy_side):
            """
            初始化敵方信息追蹤器
            
            參數:
                player_side: 玩家所屬陣營
                enemy_side: 敵方陣營
            """
            self.player_side = player_side  # 玩家陣營標識
            self.enemy_side = enemy_side    # 敵方陣營標識
            self.enemy_alive = {}           # 記錄每個敵方單位是否存活，格式 {單位名稱: 存活狀態(1/0)}
            self.initial_enemy_count = 0    # 初始敵方單位數量
            self.enemy_alive_count = 0      # 當前存活敵方單位數量
            self.prev_enemy_alive_count = 0 # 前一時刻存活敵方單位數量
            self.order = []                 # 敵方單位的順序列表，保持一致的處理順序

        def init_episode(self, features):
            """
            初始化一個新的回合
            
            參數:
                features: 當前環境的觀察資料
            """
            # 初始化敵方單位存活狀態
            self.enemy_alive = {u.Name: 1 for u in features.units[self.enemy_side]}
            self.enemy_alive_count = len(self.enemy_alive)
            self.prev_enemy_alive_count = len(self.enemy_alive)
            
            # 如果順序列表為空，則初始化
            if not self.order:
                self.initial_enemy_count = len(self.enemy_alive)
                self.order = [u.Name for u in features.units[self.enemy_side]]

        def get_enemy_found(self, features):
            """
            檢測是否有敵方單位被發現
            
            參數:
                features: 當前環境的觀察資料
                
            返回:
                1表示發現敵方單位，0表示未發現
            """
            return 1 if len(features.contacts[self.player_side]) > 0 else 0

        def update_alive(self, features):
            """
            更新敵方單位的存活狀態
            
            參數:
                features: 當前環境的觀察資料
            """
            # 獲取當前敵方單位ID
            current_ids = {u.Name for u in features.units[self.enemy_side]}
            
            # 更新存活狀態
            for name, alive in list(self.enemy_alive.items()):
                if alive == 1 and name not in current_ids:
                    self.enemy_alive[name] = 0  # 標記為已消滅
            
            # 處理新出現的單位
            for name in current_ids:
                if name not in self.enemy_alive:
                    self.enemy_alive[name] = 1  # 標記為存活
                    
            # 更新存活計數
            self.enemy_alive_count = sum(self.enemy_alive.values())

        def alive_ratio(self):
            """
            計算敵方單位的存活比例
            
            返回:
                存活的敵方單位數量與初始數量的比值
            """
            return (sum(self.enemy_alive.values()) / self.initial_enemy_count) if self.initial_enemy_count > 0 else 0.0

    class FriendlyInfo:
        """
        友方單位的信息追蹤類
        用於跟踪友方單位的存活狀態和順序
        """
        def __init__(self, player_side, manager_unit_name: str):
            """
            初始化友方信息追蹤器
            
            參數:
                player_side: 玩家所屬陣營
                manager_unit_name: 管理者單位名稱
            """
            self.player_side = player_side
            self.manager_unit_name = manager_unit_name
            self.order = [manager_unit_name]     # 只包含自己（一艘船一個MyAgent）
            self.alive = {manager_unit_name: 1}  # 只包含自己

        def init_episode(self, features):
            """
            初始化一個新的回合
            
            參數:
                features: 當前環境的觀察資料
            """
            # ❶ 用名稱排序，確保友軍順序固定穩定
            if not hasattr(self, '_order_initialized') or not self._order_initialized:
                all_friend_names = sorted([u.Name for u in features.units[self.player_side]])
                self.order = all_friend_names  # 覆蓋原本只有自己的order
                self._order_initialized = True
            # 初始化存活狀態
            self.alive = {name: 1 for name in self.order}

        def update_alive(self, features):
            """
            更新友方單位的存活狀態
            
            參數:
                features: 當前環境的觀察資料
            """
            # 獲取當前友方單位ID
            current_ids = {u.Name for u in features.units[self.player_side]}
            
            # 更新存活狀態
            for name, alive in list(self.alive.items()):
                if alive == 1 and name not in current_ids:
                    self.alive[name] = 0  # 標記為已消滅
                elif alive == 0 and name in current_ids:
                    self.alive[name] = 1  # 標記為重新出現

        def alive_mask(self):
            """
            獲取友方單位的存活掩碼
            
            返回:
                一個列表，表示每個單位的存活狀態(1表示存活，0表示已消滅)
            """
            return [self.alive.get(n, 0) for n in self.order]

    def __init__(self, player_side, enemy_side=None, manager_unit_name=None):
        """
        初始化 Agent。
        
        參數:
            player_side: 玩家所屬陣營
            enemy_side: 敵方陣營，可選
            manager_unit_name: 管理者單位名稱，可選
        """
        super().__init__(player_side)
        self.player_side = player_side
        self.enemy_side = enemy_side or 'C'
        self.manager_unit_name = manager_unit_name
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(f"MyAgent")
        self.logger.setLevel(logging.INFO)
        
        # 初始化 Mylib 並傳入 self
        self.mylib = Mylib(self)
        
        # FeUdal网络参数
        class Args:
            """內部類，用於定義網絡結構和訓練參數"""
            def __init__(self):
                self.hidden_dim = 64          # 隱藏層維度
                self.n_agents = 1             # 代理數量
                self.friend_num = 4           # 其餘友艦數          ✨ NEW
                self.enemy_num = 5            # 敵人數量
                self.n_actions = 4            # 動作空間大小: 前進、左轉、右轉、攻擊
                self.goal_dim = 3             # 目標向量維度（用於FeUdal網絡）
                
                self.manager_hidden_dim = 64  # 管理者網絡隱藏層維度
                self.worker_hidden_dim = 64   # 工作者網絡隱藏層維度
                self.state_dim_d = 3          # 狀態降維後的維度
                self.embedding_dim_k = 64     # 嵌入向量維度
                self.worker_ext_coeff = 0.5     # ➤ 新增：Worker 想看到多少外在
                self.gamma = 0.99          # already there
                self.lam = 0.95          # ☆ 新增 (給 GAE 用)
                self.int_coeff = 0.1     # 內在獎勵縮放 α

        self.args = Args()
        
        # 定義兩種不同的輸入維度
        self.local_input_size = 7 + 5*self.args.friend_num + 4*self.args.enemy_num  # 7 + 5×4 + 4×5 = 47
        self.global_input_size = 5*(1+self.args.friend_num) + 4*self.args.enemy_num  # 5×5 + 4×5 = 45

        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # 修改記憶存儲方式，使用列表存儲完整的episode
        # self.episode_memory = []           # 存儲當前episode的經驗
        # self.rollouts = []                # 存儲已完成的episodes
        # self.max_episodes = 32             # 最多保存的episode數量


        # 基本超參數
        self.gamma = 0.99                  # 折扣因子
        self.lr = 5e-4                     # 學習率
        self.batch_size = 32               # 批次大小 (B)
        self.sequence_len = 16             # 序列長度 (T)，用於DRQN訓練
        # self.train_interval = 50         # 每隔多少 steps 學習一次
        self.update_freq = 10000           # 每隔多少 steps 同步 target network

        
        # self.done_condition = 0.15          # 任務完成的閾值條件
        self.done_condition = 0.2          # 任務完成的閾值條件
        self.max_distance = 90.0           # 最大距離限制
        self.win_reward = 150              # 獲勝獎勵
        self.min_win_reward = 50           # 最小獲勝獎勵
        self.reward_scale = 25             # 獎勵縮放因子
        self.loss_threshold = 1.0          # 當 loss 超過此閾值時輸出訓練資料
        self.loss_log_file = 'large_loss_episodes.txt'  # 記錄異常 loss 的 episode 到文字檔

        # 初始化兩個網絡
        self.manager = Feudal_ManagerAgent(self.global_input_size, self.args).to(self.device)
        self.worker = Feudal_WorkerAgent(self.local_input_size, self.args).to(self.device)

        # 初始化兩個優化器
        self.manager_optimizer = torch.optim.Adam(self.manager.parameters(), lr=3e-4)
        self.worker_optimizer = torch.optim.Adam(self.worker.parameters(), lr=3e-4)

        # 初始化隱藏狀態
        self.manager_hidden = self.manager.init_hidden(batch_size=1)     # batch_size 預設 1
        self.worker_hidden = self.worker.init_hidden(batch_size=1)

        # 初始化時間抽象參數
        self.c_steps = 10  # 每c步更新一次goal
        self.step_cnt = 0  # 當前步數計數器
        self.goal_t = None  # 當前目標向量

        # --- Manager 學習用 ---
        self.manager_episode_memory = []      # 存一局裡 manager 的轉移
        self.manager_rollouts = []         # 完成回合後挪到這裡
        self.prev_global_state = None         # s_t  (45 維 np.ndarray)
        self.best_distance = 1000000       # 記錄最佳距離
        self.worst_distance = 0            # 記錄最差距離
        self.total_reward = 0              # 累計總獎勵
        self.prev_score = 0                # 上一步的分數

        # 初始化緩衝區
        self.total_steps = 0               # 總步數計數器

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None             # 上一步的狀態
        self.prev_action = None            # 上一步的動作
        self.alive = None                  # 存活狀態

        # 初始化訓練統計記錄器
        self.stats_logger = Logger()
        
        # 添加遊戲結束標記
        self.episode_init = True           # 是否為回合初始化
        self.episode_step = 0              # 當前回合步數
        self.episode_count = 0             # 回合計數
        self.max_episode_steps = 2000       # 最大回合步數
        self.min_episode_steps = 100       # 最小回合步數
        self.episode_done = False          # 回合是否結束
        self.episode_reward = 0            # 當前回合累計獎勵
        self.done = False                  # 是否完成任務
        self.timed_out = False             # 是否超時
        self.frozen = False                # 船只冻结標誌（到達目標後停止行動）
        
        # ➜ 在還沒看到任何己方船之前，視為「等待生成」
        self.waiting_for_units = True        # 等全部己方船生出來
        self.waiting_for_my_unit = True      # 只等自己那一艘也行
        self.unit_spawn_tick = -1            # 用來記錄出現的時間步

        self.step_times_rl = []            # 記錄RL步驟耗時
        self.reset_cmd = ""                # 重置命令
        # 新增：追蹤每個 episode 的統計以計算 5 期平均
        self.episode_steps_history = []    # 回合步數歷史
        self.episode_loss_history = []     # 回合損失歷史
        self.episode_return_history = []   # 回合回報歷史
        
        # --- 訓練統計 ---
        self.episode_final_distance_history = []   # ⬅︎ 新增

        # Worker scale EMA 參數
        self.wk_scale_ema = 1.0            # 初始 scale 值
        self.wk_alpha = 0.01               # EMA 更新率

        # --- episode 統計（留給 FleetController） ---
        self.last_episode_step    = 0
        self.last_episode_loss    = 0.0
        self.last_episode_return  = 0.0
        self.last_episode_avg_dist = 0.0

        # __init__ 
        self.enemy_info = MyAgent.EnemyInfo(self.player_side, self.enemy_side)  # 敵方信息追蹤器
        self.friendly_info = MyAgent.FriendlyInfo(
            self.player_side,
            self.manager_unit_name
        )

        # Manager 的 episode buffer
        self.manager_episode_memory = []
        self.prev_global_state = None

        # 完整一局資料的容器
        # self.rollouts = []          # 最近幾局的 Worker transition list
        self.manager_rollouts = []  # 最近幾局的 Manager segment list
        # self.max_episodes = 4       # 只保留最近 4 局避免吃記憶體

        # Worker 段落資料
        self.seg_buffer: list[dict] = []        # 當前 10 步的 Worker/Manager transition
        self.ext_reward_sum = 0.0          # 全局外在累積
        self.beta = self.args.worker_ext_coeff
        self.prev_goal_vec = None   # 上一個 g_t（用於 intrinsic）

        # 記錄最佳/最差距離
        self.best_distance = 1000000

        # self.episode_memory = []
        self.manager_episode_memory = []
        self.traj_buffer: list[list[dict]] = []      # ⇠ 每局存一條 trajectory
        # self.full_segments: list[dict] = []          # ⇠ 完整 segment 資料
        
        # 初始化 Manager 和 Worker 網路

        # 初始化 stats logger
        # self.stats_logger = StatsLogger()

        # --- 常數（放一次就好）
        self.MANAGER_ID   = 0              # manager 視為第 0 個 agent
        self.WORKER_BASE  = 1              # worker 從 1 … A
        self.AGENTS_TOTAL = self.args.n_agents + 1

        # --- 計數器 ---
        self.ext_reward_sum = 0.0                 # (舊) 全局用，可保留不用
        self.ext_reward_agent = np.zeros(self.args.n_agents, dtype=np.float32)  # ★新增：每艦累積

    def get_unit_info_from_observation(self, features: Multi_Side_FeaturesFromSteam, side: str, unit_name: str) -> Unit:
        """
        從觀察中獲取指定單位的資訊。
        
        參數:
            features: 當前環境的觀察資料
            side: 單位所屬陣營
            unit_name: 單位名稱
            
        返回:
            指定單位的詳細資訊，如不存在則返回None
        """
        units = features.units[side]
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None
    
    def get_contact_info_from_observation(self, features: Multi_Side_FeaturesFromSteam, side: str, contact_name: str) -> dict:
        """
        從觀察中獲取指定接觸點（敵方單位）的資訊。
        
        參數:
            features: 當前環境的觀察資料
            side: 觀察接觸點的陣營
            contact_name: 接觸點名稱
            
        返回:
            指定接觸點的詳細資訊，如不存在則返回None
        """
        contacts = features.contacts[side]
        for contact in contacts:
            if contact['Name'] == contact_name:
                return contact
        return None

    def get_done(self, my_state: np.ndarray, unit_alive: bool) -> bool:
        """
        判斷當前代理管理的船只是否完成任務
        
        參數:
            my_state: 當前船只的狀態向量
            unit_alive: 船只是否存活
            
        返回:
            布爾值，表示該船只是否完成任務
        """
        # ① 前 n 步或船未 spawn 都不判定
        if self.episode_step < 3 or self.waiting_for_units or (not unit_alive):
            return False
        # ② 超時
        if self.episode_step >= self.max_episode_steps:
            self.timed_out = True
            return True
        # ③ 抵達
        return my_state[0] < self.done_condition

    def get_distance(self, dx, dy):
        """
        计算智能体与目标之间的距离，支持NumPy数组和PyTorch张量
        
        參數:
            dx: x方向上的相對距離
            dy: y方向上的相對距離
            
        返回:
            智能體與目標之間的歐氏距離
        """
        if isinstance(dx, torch.Tensor):
            # 使用PyTorch操作
            return torch.sqrt((dx)**2 + (dy)**2)
        else:
            # 使用NumPy操作
            return np.sqrt((dx)**2 + (dy)**2)
            
    def action(self, features: Multi_Side_FeaturesFromSteam) -> str:
        """
        根據觀察到的特徵執行動作。
        這是智能體的主要決策函數，每一步都會被呼叫。
        
        參數:
            features: 當前環境的觀察資料，包含單位、接觸點等信息
            
        返回:
            執行的動作命令（字串），將被發送到遊戲引擎
        """
        # 冻结检查：如果已经到达目标，直接返回空指令
        if getattr(self, "frozen", False):
            return ""          # 回傳空字串，什麼也不做
            
        # 記錄「前一步」transition索引
        last_step_idx = len(self.seg_buffer) - self.args.n_agents   # 指向上一時間步那 A 筆 worker

        if self.episode_init:
            # 第一次執行 action()，初始化敵人與友軍資訊
            self.enemy_info.init_episode(features)
            self.friendly_info.init_episode(features)
            
            # 找到Manager單位在友軍列表中的位置
            self.manager_idx = None
            for i, name in enumerate(self.friendly_info.order):
                if name == self.manager_unit_name:
                    self.manager_idx = i
            
            if self.manager_idx is None:
                self.logger.warning(
                    f"manager_unit_name={self.manager_unit_name} 不在友艦清單內，"
                    "Manager 仍會運作，但無法對應特定艦艇。"
                )
            
            self.episode_init = False
            self.episode_count += 1
            self.logger.info(f"episode: {self.episode_count}")
            
            # self.logger.info(f"DEBUG: 初始我方從 features 拿到 = {[u.Name for u in features.units[self.player_side]]}")
            # self.logger.info(f"DEBUG: 初始敵方從 features 拿到 = {[u.Name for u in features.units[self.enemy_side]]}")

        if features.sides_[self.player_side].TotalScore == 0:
            self.prev_score = 0
        # print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        has_unit = any(True for _ in features.units[self.player_side])

        # (A) 還在等待重生 → 什麼都不做
        if not has_unit and self.waiting_for_units:
            return ""          # 等下一個 tick

        # (B) 真正全滅才算失敗
        if not has_unit and not self.waiting_for_units:
            self.logger.warning("所有己方單位均已消失，判定失敗")
            self.done = True
            return ""
            
        # 第一次偵測到船時把等待旗標關掉
        if has_unit and self.waiting_for_units:
            self.waiting_for_units = False
            self.unit_spawn_tick = self.total_steps   # 只是做個記錄
            
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        local_states_all = self.get_states(features)                # 完整觀測，長度=友艦數, 每項 47
        # 建議：用 manager_idx 取出正確那艘
        if self.manager_idx is not None and self.manager_idx < len(local_states_all):
            local_states_rl = [local_states_all[self.manager_idx]]     # 給RL網路用，只取自己那艘
        else:
            local_states_rl = [local_states_all[0]]                    # fallback: 取第一艘
        local_states = local_states_rl                              # 供後面 Worker 部分使用
        global_state = self.get_global_state(features)              # 長度 45

        # 預設獎勵，確保大小 ≥1
        rewards = [0.0] * max(1, len(self.friendly_info.order))

        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            # 檢查是否達到結束條件 - 只看自己的狀態
            if self.manager_idx is not None and self.manager_idx < len(local_states_all):
                my_state = local_states_all[self.manager_idx]
            else:
                my_state = local_states_all[0]  # fallback
            
            # 獲取自己船只的存活狀態
            my_unit = self.get_unit_info_from_observation(features, self.player_side, self.manager_unit_name)
            unit_alive = my_unit is not None
            self.done = self.get_done(my_state, unit_alive)
            
            # 真正計算本步的獎勵
            rewards = self.get_rewards(features,
                                   self.prev_state,
                                   local_states_all,
                                   features.sides_[self.player_side].TotalScore)
            
            # 計算平均獎勵（僅用於 debug）
            rewards_arr = np.array(rewards, dtype=np.float32)
            avg_reward = rewards_arr.mean() if rewards_arr.size > 0 else 0.0
            self.total_reward += avg_reward
            self.logger.debug(f"step reward = {avg_reward:.3f}")
            
            # 累積外在獎勵（全局）- 不再使用，改用per-agent版本
            ext_r_step = float(np.sum(rewards))        # scalar
            # self.ext_reward_sum += ext_r_step  # ← 不再使用全局累積

            # 即時計算內在 rᴵ，寫回「上一個 worker transition」
            if self.prev_goal_vec is not None and self.prev_global_state is not None:
                delta_s_d = (global_state - self.prev_global_state)[1:4]   # dist, sin, cos
                g_vec      = torch.tensor(self.prev_goal_vec, device=self.device, dtype=torch.float32) # (A,3)
                d_vec      = torch.tensor(delta_s_d,      device=self.device, dtype=torch.float32)     # (3,)
                r_intr_vec = self.compute_intrinsic_reward(g_vec, d_vec).cpu().numpy()                # (A,)

                # ── 寫回剛才計的「前一步」──
                # ★ 新增保險條件：不足就直接跳過回填
                if last_step_idx < 0 or len(self.seg_buffer) < (last_step_idx + self.args.n_agents):
                    pass        # 什麼也不做；下次再補
                else:
                    # ★ 累積個別外在獎勵
                    self.ext_reward_agent += np.array(rewards[:self.args.n_agents], dtype=np.float32)
                    # self.ext_reward_sum += ext_r_step   # ← 保留也行，但個別版不再使用
                    
                    α = self.args.int_coeff
                    for a in range(self.args.n_agents):
                        indiv_ext = rewards[a]                      # ★單艦外在
                        t = self.seg_buffer[last_step_idx + a]

                        # 內在 + 外在（仍用 β 調外在權重）
                        t["reward"]    = α * r_intr_vec[a] + self.beta * indiv_ext
                        t["r_int_raw"] = float(r_intr_vec[a])
                        t["r_ext_raw"] = float(indiv_ext)
        
        # 把 cost 寫回「上一個 Manager transition」
        if (len(self.seg_buffer) >= (self.args.n_agents + 1)
            and self.prev_goal_vec is not None
            and self.prev_global_state is not None):
            
            mgr_idx = -(self.args.n_agents + 1)
            delta_s_d = (global_state - self.prev_global_state)[1:4]

            g_vec = torch.tensor(self.prev_goal_vec, dtype=torch.float32, device=self.device)
            d_vec = torch.tensor(delta_s_d,      dtype=torch.float32, device=self.device)

            cost = float(torch.cosine_similarity(
                         g_vec.mean(dim=0, keepdim=True),
                         d_vec.unsqueeze(0)).item())
            self.seg_buffer[mgr_idx]["cost"] = cost
            
            # 將經驗添加到當前episode的記憶中
            # self.episode_memory.append((global_state, local_states, self.prev_action, rewards, self.done, self.alive))
            
            # 檢查遊戲是否結束
            if self.done or self.episode_step > self.max_episode_steps:
                self.episode_done = True
                self.logger.info(f"遊戲結束! 總獎勵: {self.episode_reward:.4f}")
                
                # 處理最後一段不足 10 步的 roll-out
                if self.prev_goal_vec is not None and len(self.seg_buffer) > 0:
                    # 1) intrinsic reward
                    delta_s = global_state - self.prev_global_state          # (45,)
                    delta_s_d = delta_s[1:4]               # dist, sin, cos

                    g_vec = torch.tensor(self.prev_goal_vec, device=self.device, dtype=torch.float32)   # (A,3)
                    d_vec = torch.tensor(delta_s_d,      device=self.device, dtype=torch.float32)       # (3,)

                    # (A,)  每艘船自己的 intrinsic reward
                    r_intr_vec = self.compute_intrinsic_reward(g_vec, d_vec)    # tensor, shape (A,)

                    # 計算獎勵 - 累積per-agent獎勵
                    self.ext_reward_agent += np.array(rewards[:self.args.n_agents], dtype=np.float32)
                    ext_r_step = float(np.sum(rewards))   # ➤ 全局外在（scalar）
                    # self.ext_reward_sum += ext_r_step  # ← 不再使用全局累積

                    # 每 c_steps 產生新目標，並把舊段落推進 buffer
                    if self.episode_step % self.c_steps == 0:
                        # 回填 reward
                        if len(self.seg_buffer) < (self.args.n_agents + 1):
                            pass  # 資料不足，先跳過回填
                        else:
                            r_ext_bar_agent = self.ext_reward_agent / self.c_steps    # shape (A,)

                            for t in self.seg_buffer:
                                if t["type"] == "wrk":
                                    aid = t["agent_id"]
                                    t["reward"] = ( self.args.int_coeff * t["r_int_raw"]
                                                    + self.beta * r_ext_bar_agent[aid] )
                                    t["r_ext_raw"] = float(r_ext_bar_agent[aid])
                                else:                     # Manager
                                    t["reward"] = float(r_ext_bar_agent.mean())       # Manager 仍用平均

                            self.ext_reward_agent[:] = 0.0               # ➤ 清零，開始累下一段

                    # 3) 組成 segment；存到 buffers
                    segment = {
                        "mgr": (self.prev_goal_vec.copy(),
                               self.prev_global_state.copy(),
                               delta_s.copy(),
                               float(self.ext_reward_agent.mean())),    # ← 改成平均或總和
                        "wk": self.seg_buffer.copy()
                    }
                    self.manager_rollouts.append(segment)

                    # ✅ 把 worker 的整段 trajectory 先推進 on-policy buffer
                    self.traj_buffer.append(self.seg_buffer.copy())

                    # 若仍想保有 replay（off-policy）路徑，可選擇性保留
                    # self.rollouts.extend(self.seg_buffer)

                    # 清空
                    self.seg_buffer.clear()

                # 將完成的episode添加到已完成episodes列表中
                # if len(self.episode_memory) > 0:
                #     self.rollouts.append(self.episode_memory)
                #     if len(self.rollouts) > self.max_episodes:
                #         self.rollouts.pop(0)
                #     self.episode_memory = []      # 清空等待下一局

                # Manager 也收尾
                if len(self.manager_episode_memory) > 0:
                    # 最後一段也存起來
                    if self.prev_global_state is not None:
                        delta_s = global_state - self.prev_global_state
                        self.manager_episode_memory.append((
                            self.prev_global_state.copy(),
                            delta_s
                        ))
                    self.manager_rollouts.append(self.manager_episode_memory)
                    # if len(self.manager_rollouts) > self.max_episodes:
                    #     self.manager_rollouts.pop(0)
                    self.manager_episode_memory = []      # 清空等待下一局
                    self.prev_global_state = None

                # 在遊戲結束時進行訓練
                # 回填最後一段的 reward
                if self.prev_goal_vec is not None and self.prev_global_state is not None:
                    delta_s = global_state - self.prev_global_state          # (45,)
                    delta_s_d = delta_s[1:4]               # dist, sin, cos
                    
                    g_vec = torch.tensor(self.prev_goal_vec, device=self.device, dtype=torch.float32)   # (A,3)
                    d_vec = torch.tensor(delta_s_d,      device=self.device, dtype=torch.float32)       # (3,)
                    
                    if len(self.seg_buffer) < (self.args.n_agents + 1):
                        return ""  # 資料不足，提前返回
                    
                    r_intr_vec = self.compute_intrinsic_reward(g_vec, d_vec)        # (A,)
                    r_ext_bar_agent = self.ext_reward_agent / self.c_steps

                    for t in self.seg_buffer:
                        if t["type"] == "wrk":
                            aid = t["agent_id"]
                            t["reward"] = ( self.args.int_coeff *
                                            r_intr_vec[aid].item() / self.c_steps +
                                            self.beta * r_ext_bar_agent[aid] )
                            t["r_ext_raw"] = float(r_ext_bar_agent[aid])
                        else:
                            t["reward"] = float(r_ext_bar_agent.mean())

                self.ext_reward_agent[:] = 0.0               # ➤ 清零，開始累下一段

                # 確保最後一段被推進 traj_buffer
                if len(self.seg_buffer) > 0:
                    self.traj_buffer.append(self.seg_buffer.copy())

                # episode 結束後
                if len(self.traj_buffer) == 0 and len(self.seg_buffer) > 0:
                    # ❷ 還有殘留 transition：強制 flush 成一條 trajectory
                    self.traj_buffer.append(self.seg_buffer.copy())
                    self.seg_buffer.clear()

                loss = 0.0
                if self.traj_buffer:
                    loss = self.update_actor_critic(self.traj_buffer.pop(0))
                self.logger.info(f"A2C loss={loss:.4f}")     # ← 只會印一次
                
                # ➤ 計算「存活船艦平均距離」
                avg_final_dist = self._compute_final_distance_avg(features)
                if avg_final_dist is not None:                  # 有船活著才紀錄
                    self.episode_final_distance_history.append(avg_final_dist)
                    self.logger.info(f"final_avg_dist={avg_final_dist:.2f} km")
                
                # 先把本局統計記下來，再 reset
                self.episode_steps_history.append(self.episode_step)
                self.episode_loss_history.append(loss)     # ← 用真正的 loss
                self.episode_return_history.append(self.episode_reward)

                # ★★★ 把本局統計留給 Fleet ★★★
                self.last_episode_step     = self.episode_step
                self.last_episode_loss     = loss
                self.last_episode_return   = self.episode_reward
                self.last_episode_avg_dist = avg_final_dist if avg_final_dist is not None else 0.0

                # 補齊 dummy Worker 確保陣列對齊
                if self.done and len(self.seg_buffer) % (self.args.n_agents+1) == 1:
                    # 只剩 Manager，補一筆 dummy Worker
                    self.seg_buffer.append({
                        "type":   "wrk",
                        "state_g": None,
                        "state_l": np.zeros(self.local_input_size, dtype=np.float32),
                        "goal":    np.zeros(self.args.goal_dim, dtype=np.float32),
                        "action":  -1,
                        "logp":    0.0,
                        "value":   0.0,
                        "reward":  0.0,
                        "done":    True,
                        "cost":    0.0,
                        "agent_id": 0
                    })

                # ------ 不再由單艦重置 ------
                # 內部參數晚點由 FleetController→soft_reset() 處理
                
                # ❷ 首次完成 → 凍結
                if self.done and not self.frozen:
                    self.frozen = True           # 之後這艘船 action() 都早退
                    return ""                    # 完成本次最後一條指令
                
                return ""                 # 只要回傳空字串
                if self.episode_count % 5 == 0:

                    # 計算最近 5 個 episode 的平均值
                    window = 5
                    count = len(self.episode_steps_history)
                    avg_steps = sum(self.episode_steps_history[-window:]) / min(window, count)
                    avg_loss = sum(self.episode_loss_history[-window:]) / min(window, count)
                    avg_return = sum(self.episode_return_history[-window:]) / min(window, count)
                    # 記錄平均值
                    self.stats_logger.log_stat("episode_step", float(avg_steps), self.total_steps)
                    self.stats_logger.log_stat("loss", float(avg_loss), self.total_steps)
                    self.stats_logger.log_stat("episode_return", float(avg_return), self.total_steps)

                    if len(self.episode_final_distance_history) > 0:
                        avg_final_dist = (
                            sum(self.episode_final_distance_history[-window:]) /
                            min(window, len(self.episode_final_distance_history))
                        )
                        self.stats_logger.log_stat("final_distance_avg", float(avg_final_dist), self.total_steps)
                        self.episode_final_distance_history = []   # 清掉才不會重複計

                    # 重置統計
                    self.episode_steps_history = []
                    self.episode_loss_history = []
                    self.episode_return_history = []
                
                self.done = True           # 標示自己完成
                return ""                  # 不做任何刪船動作
        
        # -------- Manager 前傳 --------
        # 確保 hidden 的第二維 = B (=1)
        if self.manager_hidden[0].size(1) != 1:
            self.manager_hidden = self.manager.init_hidden(batch_size=1)   # (1,1,64)

        mg_tensor = torch.tensor(global_state, dtype=torch.float32,
                                 device=self.device).view(1,1,-1)   # [1,1,45]
        
        # Manager：每 c_steps 產生一次新目標向量
        ### ❶ 段尾 flush  ########################################
        if self.step_cnt > 0 and (self.step_cnt % self.c_steps) == 0 \
           and len(self.seg_buffer) >= (self.args.n_agents + 1):
            r_ext_bar_agent = self.ext_reward_agent / self.c_steps    # shape (A,)
            self.seg_buffer[-(self.args.n_agents+1)]["reward"] = float(r_ext_bar_agent.mean())  # Manager用平均
            
            # --- cost calculation and backfill ---
            delta_full = global_state - self.prev_global_state
            delta_mat  = np.stack([ delta_full[aid*5+1 : aid*5+4] for aid in range(self.args.n_agents) ])  # dist, sin, cos
            goal_mat   = self.prev_goal_vec[:, : self.args.state_dim_d]      # (A,3)

            cost_vec = self.compute_intrinsic_reward(
                torch.tensor(goal_mat,  dtype=torch.float32),
                torch.tensor(delta_mat, dtype=torch.float32)
            )
            cost = float(cost_vec.mean().item())       # manager 可用平均
            self.seg_buffer[-(self.args.n_agents+1)]["cost"] = cost
            
            # 回填worker rewards
            for t in self.seg_buffer:
                if t["type"] == "wrk":
                    aid = t["agent_id"]
                    t["reward"] = ( self.args.int_coeff * t["r_int_raw"]
                                    + self.beta * r_ext_bar_agent[aid] )
                    t["r_ext_raw"] = float(r_ext_bar_agent[aid])
            
            self.ext_reward_agent[:] = 0.0
            self.traj_buffer.append(self.seg_buffer.copy())
            self.seg_buffer.clear()
            # 只訓練，不印 log
            _ = self.update_actor_critic(self.traj_buffer.pop(0))

        ### ❷ 段首：一定要有 Manager transition ###############
        # 不論 step% c_steps 是否為 0 都呼叫 self.manager(...)
        goal_seq, logp_seq, val_avg, (val_m1, val_m2), self.manager_hidden = \
            self.manager(mg_tensor, self.manager_hidden, force_same_goal=True)
        
        if (self.step_cnt % self.c_steps) == 0:
            self.goal_t = goal_seq.detach()
            self.prev_goal_vec    = self.goal_t.squeeze(0).squeeze(0).cpu().numpy()
            self.prev_global_state = global_state.copy()
        
        self.seg_buffer.append({
            "type":   "mgr",
            "state_g": global_state.copy(),                 # Manager 用
            "state_l": None,                                # Worker 才用
            "goal":    np.zeros(self.args.goal_dim, dtype=np.float32),    # ➜ (3,) 固定佔位
            "action":  None,                                # Manager 無離散動作
            "logp":    float(logp_seq[0,0,0].item()),
            "value":   float(val_avg[0,0].item()),
            "reward":  0.0,                                 # 先填 0，段落尾再補
            "done":    self.done,       # ← 原本 False，改存真實 done
            "cost":    0.0,              # 先放 0，之後回填
            "agent_id": -1          # ← 新增這一行，給個無效編號
        })

        if (self.goal_t is None):
            self.goal_t = goal_seq.detach()             # [1,1,A,3]
            self.prev_goal_vec = self.goal_t.squeeze(0).squeeze(0).cpu().numpy()
            self.prev_global_state = global_state.copy()

        self.step_cnt += 1

        # -------- Worker 前傳 --------
        wk_tensor = torch.tensor(np.asarray(local_states), dtype=torch.float32,
                               device=self.device).view(1,1,self.args.n_agents,-1)  # [1,1,1,47]
        
        with torch.no_grad():
            logits, val_avg_wk, (val_w1, val_w2), self.worker_hidden = \
                self.worker(wk_tensor, self.worker_hidden, self.goal_t)
        
        # 檢查是否有敵人
        has_enemy = len(features.contacts[self.player_side]) > 0

        # logits shape: [T=1, B=1, A, n_actions]
        A = self.args.n_agents
        n_actions = 4  # 北南西東
        # ---------- 建立 action mask（只管攻擊能不能選） ----------
        masks = []
        for i in range(A):                      # A = self.args.n_agents
            mask = torch.ones(n_actions, dtype=torch.bool, device=self.device)   # 先全部允許

            #   has_enemy = 場上是否至少偵測到 1 個 contact
            #   current_state[i][5] = 該艦「彈藥掛載比例」(0~1)
            if (not has_enemy) or (local_states[i][5] <= 0):
                mask[3] = False                 # 禁止「攻擊」（index 3）

            masks.append(mask)
        # -----------------------------------------------------------

        # 對每個 agent 采樣動作
        actions = []
        # 建立 action mask、覆蓋 logits
        logits_masked = logits.clone()           # [1,1,A,4] 
        for j in range(A):
            # 修正 1: 為 logits_wk 增加保底處理
            if not masks[j].any():
                # 如果所有動作都不能執行，至少允許前進
                masks[j][0] = True
            logits_masked[0,0,j][~masks[j]] = -1e9   # 禁止攻擊
        
        for i in range(A):
            # 對每個 agent 采樣
            dist = torch.distributions.Categorical(logits=logits_masked[0,0,i])
            # 取樣
            sampled = dist.sample()                # ⇐ 回傳的張量已在 logits 同一個 device
            act_i = int(sampled.item())          # python 整數，方便後續指令
            logp = float(dist.log_prob(sampled).item())   # OK：同一 device
            v_i = float(val_avg_wk[0,0,i].item())
            
            # 記錄 transition
            self.seg_buffer.append({
                "type":   "wrk",
                "state_g": None,
                "state_l": local_states[i].copy(),
                "goal":    self.goal_t.squeeze(0).squeeze(0)[i].cpu().numpy(),   # 不再截前 3 維
                "action":  act_i,
                "logp":    logp,
                "value":   v_i,
                "reward":  0.0,           # 之後填
                "done":    self.done,      # ← 原本 False，改存真實 done
                "cost":    0.0,           # 先放 0，之後回填
                "agent_id": i             # 需要保留這個！
            })
            
            actions.append(act_i)

        # 累加外在平均獎勵
        rewards_arr = np.array(rewards, dtype=np.float32)
        avg_reward = rewards_arr.mean() if rewards_arr.size > 0 else 0.0

        # 儲存當前狀態和動作，用於下一步計算獎勵
        self.prev_state = local_states_all
        self.prev_action = actions

        # 更新友軍存活狀態並分配動作
        self.friendly_info.update_alive(features)
        alive_mask = self.friendly_info.alive_mask()
        self.alive = np.array(alive_mask, dtype=bool)
        action_cmd = ""
        for name in self.friendly_info.order:
            if name != self.manager_unit_name:      # 只管自己那艘
                continue
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:
                continue

            # 取得自己船只的状态（用manager_idx）
            if self.manager_idx is not None and self.manager_idx < len(local_states_all):
                state_i = local_states_all[self.manager_idx]
            else:
                state_i = local_states_all[0]  # fallback

            if state_i[0] < self.done_condition:
                action_cmd += "\n" + set_unit_heading_and_speed(
                    side=self.player_side, unit_name=name, heading=unit.CH, speed=0)
            else:
                action_cmd += "\n" + self.apply_action(actions[0], unit, features)

        if self.episode_step < 10:
            for unit in features.units[self.enemy_side]:
                action_cmd += "\n" + set_unit_to_mission(
                    unit_name=unit.Name,
                    mission_name='5VS5'
                )
            self.logger.debug(f"應用動作命令: {action_cmd}")
        
        # 更新前一步資料
        self.prev_state = local_states_all
        self.prev_action = actions
        self.total_steps += 1
        self.episode_step += 1
        self.episode_reward += rewards[0]
        self.alive = alive_mask


        
        return action_cmd

    def get_global_state(self, features: Multi_Side_FeaturesFromSteam) -> np.ndarray:
        """
        獲取全局狀態向量，包含所有單位的資訊
        
        參數:
            features: 當前環境的觀察資料
            
        返回:
            包含所有單位資訊的全局狀態向量
        """
        # 先更新敵方生死資料
        self.enemy_info.update_alive(features)

        vec = []

        # 友軍（包括自己，共 friend_num+1 條）
        friend_count = 0
        for name in self.friendly_info.order:
            if friend_count >= (1 + self.args.friend_num):  # 最多 friend_num+1 條
                break
            u = self.get_unit_info_from_observation(features, self.player_side, name)
            if u is None:                          # 已沉沒
                vec.extend([0,0,0,0,0])
            else:
                dist_norm, s, c = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH)
                mount_ratio = self._calc_mount_ratio(u)
                vec.extend([1, dist_norm, s, c, mount_ratio])
            friend_count += 1
        
        # 友軍不足時補0
        while friend_count < (1 + self.args.friend_num):
            vec.extend([0,0,0,0,0])
            friend_count += 1

        # 敵軍
        for name in self.enemy_info.order[:self.args.enemy_num]:
            u = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if u is None:
                vec.extend([0,0,0,0]); continue

            dist_norm, s, c = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH)
            vec.extend([1, dist_norm, s, c])

        return np.asarray(vec, dtype=np.float32)

    def _rel_to_target(self, lon: float, lat: float, ch_deg: float):
        """回傳 (dist_norm, sinθ, cosθ)，與 local state 完全相同的算法"""
        earth_r = 6371
        tgt_lon, tgt_lat = 118.27954108343, 24.333113806906
        lon_scale = np.cos(np.radians(lat))

        dx = (tgt_lon - lon) * np.pi * earth_r * lon_scale / 180.0
        dy = (tgt_lat - lat) * np.pi * earth_r / 180.0
        dist_norm = np.sqrt(dx*dx + dy*dy) / self.max_distance

        # 目標方位角 – 自船航向  (跟 get_state() 同式)
        head_rad  = np.deg2rad(90.0 - ch_deg)
        target_ang = np.arctan2(dy, dx)
        rel_ang    = (target_ang - head_rad + np.pi) % (2*np.pi) - np.pi

        return dist_norm, np.sin(rel_ang), np.cos(rel_ang)

    def _calc_mount_ratio(self, unit: Unit) -> float:
        """計算掛架比例"""
        if unit is None or not getattr(unit, "Mounts", None):
            return 0.0
        total = 0.0
        for m in unit.Mounts:
            if m.Name not in ("Hsiung Feng II Quad", "Hsiung Feng III Quad"):
                continue
            w = m.Weapons[0] if m.Weapons else None
            if w and w.MaxQuant > 0:
                total += w.QuantRemaining / w.MaxQuant
        return total / 2.0          # 兩個掛架平均

    def _compute_final_distance_avg(self, features) -> float | None:
        """回傳仍存活船艦的平均『實際距離(km)』，若全滅回傳 None"""
        dists = []
        # 使用固定順序掃描友軍，確保穩定性
        for name in self.friendly_info.order:
            u = self.get_unit_info_from_observation(features, self.player_side, name)
            if u is None:     # 已沉沒，跳過
                continue
            dist_norm, _, _ = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH)
            dists.append(dist_norm * self.max_distance)   # 轉回 km
        return None if len(dists) == 0 else sum(dists)/len(dists)

    def get_state(self, features: Multi_Side_FeaturesFromSteam, ac: Unit) -> np.ndarray:
        """
        獲取單個單位的狀態向量
        
        參數:
            features: 當前環境的觀察資料
            ac: 要獲取狀態的單位
            
        返回:
            包含單位資訊的狀態向量
        """
        # 先更新敵方生死資料
        self.enemy_info.update_alive(features)

        # 敵人存在與存活比率
        enemy_found = 1 if features.contacts[self.player_side] else 0
        alive_ratio = self.enemy_info.alive_ratio()

        # 時間步比率
        step_ratio = self.episode_step / self.max_episode_steps

        # 基礎向量
        ac_lon, ac_lat = float(ac.Lon), float(ac.Lat)
        dist_norm, rel_sin, rel_cos = self._rel_to_target(ac_lon, ac_lat, ac.CH)
        mount_ratio = self._calc_mount_ratio(ac)

        base_state = np.array([
            dist_norm,          # 0
            rel_sin,           # 1
            rel_cos,           # 2
            enemy_found,       # 3
            alive_ratio,       # 4
            mount_ratio,       # 5
            step_ratio         # 6
        ], dtype=np.float32)

        # 友軍資訊（除了自己）
        friend_states = []
        for name in self.friendly_info.order:
            if name == ac.Name:       # 跳過自己
                continue
            u = self.get_unit_info_from_observation(features, self.player_side, name)
            if u is None:                          # 已沉沒
                friend_states.extend([0,0,0,0,0])
            else:
                dist_norm, rel_sin, rel_cos = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH)
                mount_ratio = self._calc_mount_ratio(u)
                friend_states.extend([1, dist_norm, rel_sin, rel_cos, mount_ratio])
            if len(friend_states)//5 >= self.args.friend_num:
                break
        while len(friend_states) < 5*self.args.friend_num:    # 不足補 0
            friend_states.extend([0,0,0,0,0])

        # 敵軍資訊
        enemy_states = []
        for name in self.enemy_info.order[:self.args.enemy_num]:
            u = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if u is None:
                enemy_states.extend([0, 0, 0, 0])
                continue
            
            dist_norm, rel_sin, rel_cos = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH)
            enemy_states.extend([1, dist_norm, rel_sin, rel_cos])

        # 組合所有狀態
        state = np.concatenate([
            base_state,
            np.array(friend_states, dtype=np.float32),
            np.array(enemy_states, dtype=np.float32)
        ])
        return state

    def get_states(self, features: Multi_Side_FeaturesFromSteam) -> list[np.ndarray]:
        """依 self.friendly_info.order 回傳一串 state，長度 = 友艦數"""
        states = []
        for name in self.friendly_info.order:
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:   # 已沉沒，用 0 向量佔位
                states.append(np.zeros(self.local_input_size, dtype=np.float32))
            else:
                states.append(self.get_state(features, unit))
        return states

    def get_reward(self, state: np.ndarray, next_state: np.ndarray, score: int) -> np.ndarray:
        """
        計算單個單位的獎勵值
        
        參數:
            state: 當前狀態向量
            next_state: 下一個狀態向量
            score: 當前遊戲分數
            
        返回:
            計算得到的獎勵值
        """
        # 計算全局 reward
        reward = 0
        
        # 場景score
        current_score = score
        # 場景score變化
        score_change = current_score - self.prev_score
        self.prev_score = current_score
        # 場景score變化獎勵
        reward += score_change
        # 發現敵人獎勵
        if state[3] == 0.0 and next_state[3] == 1.0:
            reward += 10
        # 往敵方探索獎勵
        reward += 200 * (state[0] - next_state[0])
        # reward += 2000 * (state[0] - next_state[0])
        # 少一個敵人+10
        # if self.enemy_info.enemy_alive_count < self.enemy_info.prev_enemy_alive_count:
        #     reward += 10 *(self.enemy_info.prev_enemy_alive_count - selfinfo.enemy_alive_count)
        # self.enemy_info.prev_enemy_alive_count = self.enemy_info.enemy_.enemy_alive_count

        # 到達目標點獎勵
        if state[0] >= self.done_condition and next_state[0] < self.done_condition:
            reward += 20

        # 任務完成獎勵
        if next_state[0] < self.done_condition and self.done:
            win_reward = self.win_reward * (1- (self.episode_step - self.min_episode_steps) / (self.max_episode_steps - self.min_episode_steps))
            win_reward = max(win_reward, self.min_win_reward)
            reward += win_reward

        # 原始獎勵
        raw_reward = reward
        # 獲勝獎勵200 + 敵軍總數 7 *擊殺獎勵 20 + 最大距離獎勵 200*7
        max_return = self.win_reward + self.enemy_info.initial_enemy_count * 20 +  100
        scaled_reward = raw_reward/(max_return/self.reward_scale)
        # self.logger.info(f"raw reward: {raw_reward:.4f}, scaled reward: {scaled_reward:.4f}")
        # 將標量 reward 擴展為多代理人向量
        # return raw_reward
        return scaled_reward
    
    def get_rewards(self,features: Multi_Side_FeaturesFromSteam, state: list[np.ndarray], next_state: list[np.ndarray], score: int) -> list[np.ndarray]:
        """
        為當前代理管理的船只計算獎勵值
        
        參數:
            features: 當前環境的觀察資料
            state: 當前所有單位的狀態向量列表
            next_state: 下一個所有單位的狀態向量列表
            score: 當前遊戲分數
            
        返回:
            只包含自己船只獎勵的列表 (長度=1，與n_agents對齊)
        """
        # 先確定 self.manager_idx 已在 action() 裡算好
        my_idx = self.manager_idx
        
        if my_idx is None or my_idx >= len(state) or my_idx >= len(next_state):
            # fallback: 如果找不到manager_idx，使用第一艘船
            my_idx = 0
            
        if my_idx < len(state) and my_idx < len(next_state):
            my_reward = self.get_reward(state[my_idx], next_state[my_idx], score)
        else:
            my_reward = 0  # 默認獎勵
            
        return [my_reward]          # 長度仍與 n_agents (=1) 對齊

    def compute_intrinsic_reward(self, goal_vec, delta_s):
        """
        計算內在獎勵
        
        參數:
            goal_vec: 目標向量
            delta_s: 狀態變化   
        返回:
            內在獎勵
        """
        # 防零向量；加 1e-8 避免除 0
        g_norm = normalize(goal_vec, dim=-1, eps=1e-8)
        d_norm = normalize(delta_s,  dim=-1, eps=1e-8)
        return (g_norm * d_norm).sum(-1)      # 等同 cosine_similarity

    def apply_action(self, action: int, ac: Unit, features: Multi_Side_FeaturesFromSteam) -> str:
        """
        將動作轉換為 CMO 命令
        
        參數:
            action: 動作ID (0=前進, 1=左轉, 2=右轉, 3=攻擊)
            ac: 執行動作的單位
            features: 當前環境的觀察資料
            
        返回:
            CMO命令字串
        """
        lat, lon = float(ac.Lat), float(ac.Lon)
        if action == 0: #前進
            heading = ac.CH
        elif action == 1: #左轉
            heading = ac.CH-30
        elif action == 2: #右轉
            heading = ac.CH+30
        elif action == 3:  # 攻擊
            # 檢查是否有彈藥
            has_ammo = False
            enemy = random.choice(features.contacts[self.player_side])
            for mount in ac.Mounts:
                name = getattr(mount, 'Name', None)
                if name not in ('Hsiung Feng II Quad', 'Hsiung Feng III Quad'):
                    continue
                        
                weapons = getattr(mount, 'Weapons', [])
                if weapons and weapons[0].QuantRemaining > 0:
                    if name == 'Hsiung Feng III Quad':
                        weapon_id = 1133
                    elif name == 'Hsiung Feng II Quad':
                        weapon_id = 1934
                    has_ammo = True
                    break
            if not has_ammo:
                # 無彈藥，保持前進
                heading = ac.CH
            else:
                # 有彈藥，執行攻擊
                return manual_attack_contact(
                    attacker_id=ac.ID,
                    contact_id=enemy['ID'],
                    weapon_id=weapon_id,
                    qty=1
                )
            
        heading %= 360          # 任何正負角度一次搞定
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=ac.Name,
            heading=heading,
            speed=30
        )
    
    # def train(self):
    #     """使用 replay buffer 訓練"""
    #     pass

    # def train_on_policy(self):
    #     """使用 on-policy 訓練 Worker"""
    #     pass

    # def train_manager(self):
    #     """使用當前 Manager 網路重新產生目標並訓練"""
    #     return 0.0

    def update_actor_critic(self, traj: list[dict]) -> float:
        """
        一條 trajectory 內含 T*(A+1) 個 transition
        第 0 個是 Manager，1…A 是 Worker
        """
        if len(traj) == 0:
            self.logger.warning("Empty trajectory - skip update.")
            return 0.0

        # 修正 2: 自動修正開始時的NaN
        for t in traj:
            if np.isnan(t.get("reward", 0.0)):
                t["reward"] = 0.0
            if np.isnan(float(t.get("done", False))):
                t["done"] = False

        device = self.device
        Ag = self.args.n_agents + 1
        # ❸ 長度不足 Ag 直接學（T=1 也 OK）
        T = len(traj) // Ag if len(traj) >= Ag else 1

        # ---------- 1. 轉 list ----------
        state_g = []; state_l = []; goal = []
        action  = []; logp = []; value = []; reward = []
        done = []; cost = []
        for t in traj:
            state_g.append(
                t["state_g"] if t["state_g"] is not None
                else np.zeros(self.global_input_size, dtype=np.float32)
            )
            state_l.append(
                t["state_l"] if t["state_l"] is not None
                else np.zeros(self.local_input_size,  dtype=np.float32)
            )
            goal.append(t["goal"])
            action.append(-1 if t["action"] is None else t["action"])   # manager=-1
            logp.append(t["logp"]); value.append(t["value"]); reward.append(t["reward"])
            done.append(t["done"]); cost.append(t["cost"])

        # --------------- 2. 重新組 batch --------------- #
        A  = self.args.n_agents          # =1
        Ag = A + 1                       # =2   (含 manager)

        # 先把 python list → numpy → tensor
        state_g_raw = torch.from_numpy(np.asarray(state_g , np.float32)).to(device).view(T,Ag,-1)
        state_l_raw = torch.from_numpy(np.asarray(state_l , np.float32)).to(device).view(T,Ag,-1)
        goal_raw    = torch.from_numpy(np.asarray(goal    , np.float32)).to(device).view(T,Ag,-1)
        action_raw  = torch.from_numpy(np.asarray(action  , np.int64 )).to(device).view(T,Ag)
        reward_raw  = torch.from_numpy(np.asarray(reward  , np.float32)).to(device).view(T,Ag)
        done_raw    = torch.from_numpy(np.asarray(done    , np.float32)).to(device).view(T,Ag)            # 1=終結
        cost_raw    = torch.from_numpy(np.asarray(cost    , np.float32)).to(device).view(T,Ag)            # Manager 部分才有值
        
        # 保險閥：所有張量餵進 loss 前 torch.nan_to_num()
        reward_raw = torch.nan_to_num(reward_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        cost_raw   = torch.nan_to_num(cost_raw,   nan=0.0)

        # ---- Manager ----
        mgr_state   = state_g_raw[:,0,:]                   # [T,45]
        mgr_in      = mgr_state[:,None,:]   # [T,1,45]
        h0m         = self.manager.init_hidden(batch_size=1)      # (1,1,64)

        # ---- Worker ----
        wk_state    = state_l_raw[:,1:,:]                  # [T,A,47]  (去掉 manager 的佔位)
        wk_in       = wk_state[:,None,:,:]                # [T,1,A,47]
        goal_in     = goal_raw[:,1:,:][:,None,:,:]        # [T,1,A,goal_dim]
        h0w         = self.worker.init_hidden(batch_size=1)       # (1, 1, 64)

        # 檢查 hidden 尺寸是否正確
        if self.manager_hidden[0].size(1) != 1:
            self.manager_hidden = self.manager.init_hidden(batch_size=1)
        if self.worker_hidden[0].size(1) != self.args.n_agents:
            self.worker_hidden = self.worker.init_hidden(batch_size=1)

        # --------------- 3. 前向傳遞 --------------- #
        _, _, val_avg_mgr, (val_mgr1, val_mgr2), _ = self.manager(mgr_in, h0m)
        val_mgr = val_avg_mgr.squeeze(1)          # ➜ [T]

        logits_wk, val_avg_wk, (val_wk1, val_wk2), _ = self.worker(wk_in, h0w, goal_in)
        # logits_wk: [T,1,A,4] → 去掉 batch=1
        logits_wk = logits_wk.squeeze(1)                  # [T,A,4]
        val_wk    = val_avg_wk.squeeze(1).squeeze(-1)         # [T,A]

        # ---------- 5. GAE advantage calculation ----------
        gamma = self.gamma            # 0.99
        
        # 提取 Manager 的資料
        rewards_mgr = reward_raw[:,0]                    # [T]
        
        # --- Manager delta & GAE ---
        delta_mgr = rewards_mgr + gamma * \
                    torch.cat([val_mgr[1:], torch.zeros_like(val_mgr[:1])]) * \
                    (1 - done_raw[:,0]) - val_mgr      # [T]

        adv_mgr = compute_gae(delta_mgr, gamma, self.args.lam, done_raw[:,0])
        return_mgr = adv_mgr + val_mgr                 # for value loss

        # --- Manager policy loss (ONLY adv * cost) ---
        cost_vec = cost_raw[:,0]                         # [T]
        loss_act_mgr = -(adv_mgr.detach() * cost_vec).mean()

        # --- Worker delta & GAE ---
        # 先湊好 next value
        next_val_wk = torch.cat([val_wk[1:], torch.zeros_like(val_wk[:1])], dim=0)

        delta_wk = reward_raw[:,1:] + gamma * next_val_wk * (1 - done_raw[:,1:]) - val_wk
        adv_wk   = compute_gae(delta_wk, gamma, self.args.lam, done_raw[:,1:])
        return_wk = adv_wk + val_wk

        # --- Worker policy loss ---
        dist = torch.distributions.Categorical(logits=logits_wk)
        logp_wk = dist.log_prob(action_raw[:,1:])
        # 把 ±∞ → 0，NaN → 0
        logp_wk = torch.nan_to_num(logp_wk, nan=0.0, neginf=0.0, posinf=0.0)      # ★新增
        entropy = dist.entropy().mean()
        entropy = torch.nan_to_num(entropy, nan=0.0, neginf=0.0, posinf=0.0)      # ★新增

        loss_act_wk = -(adv_wk.detach() * logp_wk).mean()

        # 修正 2: 自動修正而非跳過
        for tensor_name, tensor in [("adv_mgr", adv_mgr), ("adv_wk", adv_wk), ("logp_wk", logp_wk), ("val_mgr", val_mgr), ("val_wk", val_wk)]:
            if torch.isnan(tensor).any():
                self.logger.warning(f"NaN detected in {tensor_name}, auto-correcting to 0.")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- Manager value loss (two heads) ---
        val_mgr1 = val_mgr1.squeeze(1)      # [T]
        val_mgr2 = val_mgr2.squeeze(1)
        
        # 修正 3: clamp loss 裡面異常值
        val_mgr1 = torch.nan_to_num(val_mgr1, nan=0.0)
        val_mgr2 = torch.nan_to_num(val_mgr2, nan=0.0)
        
        # 計算變異數比例自動對齊
        var_wk  = adv_wk.pow(2).mean().detach()
        var_mgr = adv_mgr.pow(2).mean().detach()
        mgr_scale = torch.sqrt((var_wk + 1e-8) / (var_mgr + 1e-8))
        
        # 1️⃣ 先做安全截斷，避免 0/0 或過大 - 自動修正
        if torch.isnan(mgr_scale) or torch.isinf(mgr_scale):
            self.logger.warning("mgr_scale is NaN/Inf, auto-correcting to 1.0.")
            mgr_scale = torch.ones_like(mgr_scale)  # 自動修正為1.0而非早退
        mgr_scale = torch.clamp(mgr_scale, 0.1, 10.0)  # 對齊 1e1 級別
        
        loss_val_mgr = 0.25 * (
            (mgr_scale * (return_mgr - val_mgr1)).pow(2).mean() +
            (mgr_scale * (return_mgr - val_mgr2)).pow(2).mean()
        )

        # --- Worker value loss (two heads) ---
        val_wk1 = val_wk1.squeeze(1).squeeze(-1)   # [T, A]
        val_wk2 = val_wk2.squeeze(1).squeeze(-1)
        
        # 修正 3: clamp loss 裡面異常值
        val_wk1 = torch.nan_to_num(val_wk1, nan=0.0)
        val_wk2 = torch.nan_to_num(val_wk2, nan=0.0)
        
        loss_val_wk = 0.25 * (
            (return_wk - val_wk1).pow(2).mean() +
            (return_wk - val_wk2).pow(2).mean()
        )

        # --------- 收集 raw 內／外在 ----------
        r_int_list, r_ext_list = [], []
        for t in traj:
            if t["type"] == "wrk":
                r_int_list.append(t.get("r_int_raw", 0.0))
                r_ext_list.append(t.get("r_ext_raw", 0.0))

        r_int_mean = np.mean(r_int_list) if r_int_list else 0.0
        r_ext_mean = np.mean(r_ext_list) if r_ext_list else 0.0

        # ← 在這裡印出各部分
        self.logger.info(
            f"loss_act_mgr={loss_act_mgr.item():.6f} | "
            f"loss_val_mgr={loss_val_mgr.item():.6f} | "
            f"loss_act_wk={loss_act_wk.item():.6f} | "
            f"loss_val_wk={loss_val_wk.item():.6f} | "
            f"entropy={entropy.item():.6f} | "
            f"r_int={r_int_mean:.3f} | r_ext={r_ext_mean:.3f}"
        )

        # ---------- 8. 總 loss ----------
        total_loss = (loss_act_mgr + loss_val_mgr
                      + loss_act_wk + loss_val_wk
                      - 0.01 * entropy)

        # 最後一關：自動修正而非早退
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.zeros(1, device=self.device)   # 直接歸零，不 early-return
        
        self.logger.info(f"total_loss={total_loss.item():.6f}")

        self.manager_optimizer.zero_grad()
        self.worker_optimizer.zero_grad()
        total_loss.backward()                                      # ★搬到 if 之後
        
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 5)
        self.manager_optimizer.step()
        self.worker_optimizer.step()
        
        # 3️⃣ 限制 Manager 的 log σ，避免 std 溢出 / 下溢
        with torch.no_grad():
            self.manager.logstd.clamp_(-5.0, 2.0)  # 對應 std ≈ e^-5 ~ e^2 ≈ 0.0067-7.4
        self.last_episode_loss = float(total_loss.item())
        return float(total_loss.item())

    def reset(self):
        """重置智能體狀態"""
        self.episode_init = True
        self.episode_done = False
        self.episode_reward = 0
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()
        self.goal_t = None
        self.step_cnt = 0
        self.best_distance = 1000000
        self.worst_distance = 0
        self.prev_state = None
        self.prev_action = None
        self.episode_step = 0
        self.done = False
        self.logger.info("重置遊戲狀態，準備開始新的episode")

        # 組合多個命令
        action_cmd = ""
        action_cmd = self.reset_cmd

        return action_cmd
    
    def get_reset_cmd(self, features: Multi_Side_FeaturesFromSteam):
        """
        產生重置遊戲所需的命令
        """
        return ""      # 讓 FleetController 統一處理


    def build_full_reset_cmd(self, features: Multi_Side_FeaturesFromSteam):
        """
        產生完整重置命令但不改變內部狀態
        """
        return ""      # 讓 FleetController 統一處理

    @property
    def completed(self):
        """為了兼容FleetController，提供completed屬性"""
        return self.done or self.timed_out or self.frozen

    def soft_reset(self):
        """輕量級重置，只清內部狀態，不發 Lua 指令"""
        # --- 原本就有的欄位 ---
        self.episode_init = True
        self.episode_done = False
        self.episode_reward = 0
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()
        self.goal_t = None
        self.step_cnt = 0
        self.best_distance = 1000000
        self.worst_distance = 0
        self.prev_state = None
        self.prev_action = None
        self.episode_step = 0
        self.done = False
        self.timed_out = False
        self.prev_goal_vec = None
        self.prev_global_state = None
        self.seg_buffer.clear()          # ★ 清空上一局留下的 transition
        self.ext_reward_sum = 0.0        # ★ 一併歸零
        self.ext_reward_agent[:] = 0.0   # ★ 重置per-agent累積器
        self.frozen = False              # 解除凍結
        # --- 必加：等待旗標 ---
        self.waiting_for_units = True    # 等全部己方船 spawn
        self.waiting_for_my_unit = True  # 等自己的船 spawn
        self.unit_spawn_tick = -1
        # 重新歸零統計 –– 不然下一局仍然沿用舊值
        self.last_episode_step   = 0
        self.last_episode_loss   = 0.0
        self.last_episode_return = 0.0
        self.last_episode_avg_dist = 0.0
        self.logger.info("軟重置完成")