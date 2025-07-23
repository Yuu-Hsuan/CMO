from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed, manual_attack_contact, delete_unit, add_unit, set_unit_to_mission, auto_attack_contact
import pycmo.lib.actions as actlib
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import Multi_Side_FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
from scripts.new_try.Mylib import *
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
        def __init__(self, player_side, n_agents:int):
            """
            初始化友方信息追蹤器
            
            參數:
                player_side: 玩家所屬陣營
                n_agents: 只交給前 n_agents 條船
            """
            self.player_side = player_side  # 玩家陣營標識
            self.order = []                 # 友方單位的順序列表，保持一致的處理順序
            self.alive = {}                 # 記錄每個友方單位是否存活，格式 {單位名稱: 存活狀態(1/0)}
            self.n_agents = n_agents        # 只交給前 n_agents 條船

        def init_episode(self, features, agent):
            """
            初始化一個新的回合
            
            參數:
                features: 當前環境的觀察資料
                agent: MyAgent 實例，用於存取 AOE_list
            """
            # 如果順序列表為空，則初始化
            if not self.order:
                all_names = [u.Name for u in features.units[self.player_side]]
                for name in all_names:
                    if name not in self.order:
                        self.order.append(name)
            
            # 把 AOE 剔除，只保留前 n_agents 艘船
            self.order = [n for n in self.order if n not in agent.AOE_list][:self.n_agents]
            
            # 初始化所有單位為存活狀態
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
            return [self.alive.get(n, 0) for n in self.order[:self.n_agents]]

    def __init__(self, player_side, enemy_side=None, manager_unit_name=None):
        """
        初始化 Agent。
        
        參數:
            player_side: 玩家所屬陣營
            enemy_side: 敵方陣營，可選
            manager_unit_name: 管理者單位名稱，可選
        """
        super().__init__(player_side)
        
        # === 固定參數（新增或覆寫） ===
        self.AOE_list   = ['AOE532_1']                 # 只會有 1 艘 AOE
        self.TC_PORT_LON, self.TC_PORT_LAT = 120.51182439926629, 24.285878053426778
        
        self.player_side = player_side
        self.enemy_side = enemy_side or 'China'
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
                self.n_agents = 5             # 只控制 5 艘護航艦
                self.enemy_num = 7            # 敵船 7 艘
                self.n_actions = 7            # 動作空間大小: 7個巡邏任務
                self.goal_dim = 3             # 目標向量維度（用於FeUdal網絡）
                
                self.manager_hidden_dim = 64  # 管理者網絡隱藏層維度
                self.worker_hidden_dim = 64   # 工作者網絡隱藏層維度
                self.state_dim_d = 3          # 狀態降維後的維度
                self.embedding_dim_k = 64     # 嵌入向量維度
                self.worker_ext_coeff = 0.5     # ➤ 新增：Worker 想看到多少外在
                self.gamma = 0.99          # already there
                self.lam = 0.95          # ☆ 新增 (給 GAE 用)
                self.int_coeff = 0.1     # 內在獎勵縮放 α

        # --- 1. 先建 Args 物件 ----------------------
        self.args = Args()                 # ← 放在最前面

        # --- 2. 再用它算尺寸 ------------------------
        self.A = self.args.n_agents
        self.E = self.args.enemy_num
        self.LOCAL_DIM  = 7 + 5*(self.A-1) + 4*self.E   # 55
        self.GLOBAL_DIM = 5*self.A         + 4*self.E   # 53

        # --- reward 相關常數 -------------------------------------------------
        self.win_reward   = 500          # 勝利獎勵
        self.fail_reward  = -300         # 敗北懲罰
        self.reward_scale = 25           # 獎勵縮放因子
        self.max_return   = self.win_reward + self.args.enemy_num * 60   # ≈ 300(達港) + 7*10(殲敵) + 預留其他

        # --- 追蹤 AOE 距離 ---------------------------------------------------
        # key = AOE 名稱，value = 上一步的 state 向量（np.ndarray）
        self.AOE_last_states: dict[str, np.ndarray] = {}
        self.AOE_states:      dict[str, np.ndarray] = {}
        
        # 定義兩種不同的輸入維度（已移至常數區）
        # self.local_input_size = 7 + 5*(self.args.n_agents-1) + 4*self.args.enemy_num  # 47
        # self.global_input_size = self.args.n_agents*5 + self.args.enemy_num*4         # 45

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
        self.min_win_reward = 50           # 最小獲勝獎勵
        self.loss_threshold = 1.0          # 當 loss 超過此閾值時輸出訓練資料
        self.loss_log_file = 'large_loss_episodes.txt'  # 記錄異常 loss 的 episode 到文字檔

        # 初始化兩個網絡
        self.manager = Feudal_ManagerAgent(self.GLOBAL_DIM, self.args).to(self.device)
        self.worker = Feudal_WorkerAgent(self.LOCAL_DIM, self.args).to(self.device)

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
        self.max_episode_steps = 5000       # 最大回合步數
        self.min_episode_steps = 200       # 最小回合步數
        self.episode_done = False          # 回合是否結束
        self.episode_reward = 0            # 當前回合累計獎勵
        self.done = False                  # 是否完成任務

        self.step_times_rl = []            # 記錄RL步驟耗時
        self.reset_cmd = ""                # 重置命令
        # 新增：追蹤每個 episode 的統計以計算 5 期平均
        self.episode_steps_history = []    # 回合步數歷史
        self.episode_loss_history = []     # 回合損失歷史
        self.episode_return_history = []   # 回合回報歷史
        
        # --- 訓練統計 ---
        self.episode_final_distance_history = []   # ⬅︎ 新增
        self.episode_result_history         = []   # ⬅︎ 新增：每局 1=win / 0=fail

        # Worker scale EMA 參數
        self.wk_scale_ema = 1.0            # 初始 scale 值
        self.wk_alpha = 0.01               # EMA 更新率

        # __init__ 
        self.enemy_info = MyAgent.EnemyInfo(self.player_side, self.enemy_side)  # 敵方信息追蹤器
        self.friendly_info = MyAgent.FriendlyInfo(self.player_side, self.args.n_agents)             # 友方信息追蹤器

        # Manager 的 episode buffer
        self.prev_global_state = None

        # 完整一局資料的容器
        # self.rollouts = []          # 最近幾局的 Worker transition list
        self.manager_rollouts = []  # 最近幾局的 Manager segment list
        # self.max_episodes = 4       # 只保留最近 4 局避免吃記憶體

        # Worker 段落資料
        self.seg_buffer: list[dict] = []        # 當前 10 步的 Worker/Manager transition
        self.ext_reward_sum = 0.0           # ★ 加回這行
        self.beta = self.args.worker_ext_coeff
        self.prev_goal_vec = None   # 上一個 g_t（用於 intrinsic）



        # self.episode_memory = []
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
        self.ext_reward_sum = 0.0       # ➤ 新增：累積外在獎勵

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

    # ==================== 勝負 / 結束判定 ====================
    def get_done(self, features: Multi_Side_FeaturesFromSteam,
                 state: list[np.ndarray]) -> bool:
        """
        第 0 步永遠傳 False，之後：
          • get_win == True  → done
          • get_fail == True → done
        """
        if self.episode_step == 0:
            return False
        if self.get_win(features, state):
            return True
        if self.get_fail(features, state):
            return True
        return False


    def get_win(self, features: Multi_Side_FeaturesFromSteam,
                state: list[np.ndarray]) -> bool:
        """AOE 抵達港口即勝利"""
        for name in self.AOE_list:
            # 若 AOE 狀態有更新才檢
            dist_norm = self.AOE_states.get(name, np.array([1.0]))[0]
            if dist_norm < self.done_condition:
                return True
        return False


    def get_fail(self, features: Multi_Side_FeaturesFromSteam,
                 state: list[np.ndarray]) -> bool:
        """AOE 被擊沉即失敗"""
        for name in self.AOE_list:
            if self.get_unit_info_from_observation(features, self.player_side, name) is None:
                return True
        return False

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
        # 記錄「前一步」transition索引
        last_step_idx = len(self.seg_buffer) - self.args.n_agents   # 指向上一時間步那 A 筆 worker

        if self.episode_init:
            # 第一次執行 action()，初始化敵人與友軍資訊
            self.enemy_info.init_episode(features)
            self.friendly_info.init_episode(features, self)
            
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
            
            # 如果不是 episode_init，繼續執行正常的 action 邏輯
            self.episode_count += 1
            self.logger.info(f"episode: {self.episode_count}")

            # 先把 features 存一份，給 reset() 用
            self.last_features = deepcopy(features)
            
            # 初始化 AOE_last_states
            for idx, name in enumerate(self.AOE_list):
                aoe_unit = self.get_unit_info_from_observation(
                    features, self.player_side, name)
                if aoe_unit is not None:
                    aoe_state = self.get_state(features, aoe_unit)
                    self.AOE_states[name] = aoe_state
            self.AOE_last_states = deepcopy(self.AOE_states)
            
            self.episode_init = False  # ★ 加這一行 ── 告訴程式「初始化已經做完」
            # 第 0 步就先推一遍任務
            return self.reassign_mission(features)
        
            
            
            # self.logger.info(f"DEBUG: 初始我方從 features 拿到 = {[u.Name for u in features.units[self.player_side]]}")
            # self.logger.info(f"DEBUG: 初始敵方從 features 拿到 = {[u.Name for u in features.units[self.enemy_side]]}")

        if features.sides_[self.player_side].TotalScore == 0:
            self.prev_score = 0
        # print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        has_unit = False
        for unit in features.units[self.player_side]:
            has_unit = True
        if not has_unit:
            self.logger.warning(f"找不到任何單位")
            return self.reset()  # 如果找不到單位，返回初始化
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        local_states = self.get_states(features)                # list 長度=5, 每項 47
        global_state = self.get_global_state(features)          # 長度 45

        # （放在得到 local_states 之後即可）
        for idx, name in enumerate(self.AOE_list):
            # AOE 不在 friendly_info.order 內，手動抓取
            aoe_unit = self.get_unit_info_from_observation(
                features, self.player_side, name)
            if aoe_unit is not None:
                aoe_state = self.get_state(features, aoe_unit)   # 7+… 維向量
                self.AOE_states[name] = aoe_state

        # 預設獎勵，確保大小 ≥1
        rewards = [0.0] * max(1, len(self.friendly_info.order))

        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            # 檢查是否達到結束條件
            self.done = self.get_done(features, local_states)
            
            # 真正計算本步的獎勵
            rewards = self.get_rewards(features,
                                   self.prev_state,
                                   local_states)
            
            # 計算平均獎勵（僅用於 debug）
            rewards_arr = np.array(rewards, dtype=np.float32)
            avg_reward = rewards_arr.mean() if rewards_arr.size > 0 else 0.0
            self.total_reward += avg_reward
            self.logger.debug(f"step reward = {avg_reward:.3f}")
            
            # 累積外在獎勵（全局）
            ext_r_step = float(np.sum(rewards))        # scalar
            self.ext_reward_sum += ext_r_step



            # 即時計算內在 rᴵ，寫回「上一個 worker transition」
            if self.prev_goal_vec is not None and self.prev_global_state is not None:
                delta_s_d = (global_state - self.prev_global_state)[1:4]   # dist, sin, cos
                g_vec      = torch.tensor(self.prev_goal_vec, device=self.device, dtype=torch.float32) # (A,3)
                d_vec      = torch.tensor(delta_s_d,      device=self.device, dtype=torch.float32)     # (3,)
                r_intr_vec = self.compute_intrinsic_reward(g_vec, d_vec).cpu().numpy()                # (A,)

                # ── 寫回剛才計的「前一步」──
                r_ext_share = self.beta * (ext_r_step / self.args.n_agents)   # 單步外在
                α = self.args.int_coeff
                for a in range(self.args.n_agents):
                    t = self.seg_buffer[last_step_idx + a]
                    t["reward"] = α * r_intr_vec[a] + r_ext_share
                    t["r_int_raw"] = float(r_intr_vec[a])                          # ★
                    t["r_ext_raw"] = float(ext_r_step / self.args.n_agents)        # ★
            
            # 把 cost 寫回「上一個 Manager transition」
            if len(self.seg_buffer) >= (self.args.n_agents + 1):
                mgr_idx = - (self.args.n_agents + 1)   # 最後一個 Manager 所在 index
                cost = float(torch.cosine_similarity(g_vec.mean(dim=0, keepdim=True), d_vec.unsqueeze(0)).item())
                self.seg_buffer[mgr_idx]["cost"] = cost   # cost = cos(g, Δs)
            
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

                    # 計算獎勵
                    ext_r_step = float(np.sum(rewards))   # ➤ 全局外在（scalar）
                    self.ext_reward_sum += ext_r_step

                    # 每 c_steps 產生新目標，並把舊段落推進 buffer
                    if self.episode_step % self.c_steps == 0:
                        # 回填 reward
                        β = self.args.worker_ext_coeff
                        r_ext_bar  = self.ext_reward_sum / self.c_steps                 # 段平均外在
                        r_ext_share = β * r_ext_bar / self.args.n_agents                # 均分給每艘船

                        for t in self.seg_buffer:
                            if t["type"] == "wrk":
                                aid = t["agent_id"]
                                α = self.args.int_coeff
                                t["reward"] = α * r_intr_vec[aid].item() / self.c_steps + r_ext_share
                                t["r_int_raw"] = float(r_intr_vec[aid].item() / self.c_steps)    # ★
                                t["r_ext_raw"] = float(r_ext_share)                              # ★
                            else:                               # Manager
                                t["reward"] = r_ext_bar

                        self.ext_reward_sum = 0.0               # ➤ 清零，開始累下一段

                    # 3) 組成 segment；存到 buffers
                    segment = {
                        "mgr": (self.prev_goal_vec.copy(),
                               self.prev_global_state.copy(),
                               delta_s.copy(),
                               self.ext_reward_sum),             # ← 直接存 float 就好
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
                    
                    β = self.args.worker_ext_coeff
                    r_intr_vec = self.compute_intrinsic_reward(g_vec, d_vec)        # (A,)
                    r_ext_bar  = self.ext_reward_sum / self.c_steps                 # 段平均外在
                    r_ext_share = β * r_ext_bar / self.args.n_agents                # 均分給每艘船

                    for t in self.seg_buffer:
                        if t["type"] == "wrk":
                            aid = t["agent_id"]
                            α = self.args.int_coeff
                            t["reward"] = α * r_intr_vec[aid].item() / self.c_steps + r_ext_share
                            t["r_int_raw"] = float(r_intr_vec[aid].item() / self.c_steps)    # ★
                            t["r_ext_raw"] = float(r_ext_share)                              # ★
                        else:                               # Manager
                            t["reward"] = r_ext_bar

                self.ext_reward_sum = 0.0               # ➤ 清零，開始累下一段

                # 確保最後一段被推進 traj_buffer
                if len(self.seg_buffer) > 0:
                    self.traj_buffer.append(self.seg_buffer.copy())

                # episode 結束後
                if len(self.traj_buffer) > 0:
                    loss = self.update_actor_critic(self.traj_buffer.pop(0))
                else:
                    loss = 0.0
                self.logger.info(f"A2C loss={loss:.4f}")     # ← 只會印一次
                
                # ➤ 計算「存活船艦平均距離」
                avg_final_dist = self._compute_final_distance_avg(features)
                if avg_final_dist is not None:                  # 有船活著才紀錄
                    self.episode_final_distance_history.append(avg_final_dist)
                    self.logger.info(f"final_avg_dist={avg_final_dist:.2f} km")
                
                # 記錄本局勝負結果
                is_win_episode = self.get_win(features, local_states)
                self.episode_result_history.append(1 if is_win_episode else 0)
                
                # 先把本局統計記下來，再 reset
                self.episode_steps_history.append(self.episode_step)
                self.episode_loss_history.append(loss)     # ← 用真正的 loss
                self.episode_return_history.append(self.episode_reward)

                # 重置遊戲狀態（放到最後）
                self.reset()
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

                    if len(self.episode_result_history) > 0:
                        avg_win = (
                            sum(self.episode_result_history[-window:]) /
                            min(window, len(self.episode_result_history))
                        )
                        self.stats_logger.log_stat("win_rate", float(avg_win), self.total_steps)
                        self.episode_result_history = []      # 清掉才不會重複計

                    # 重置統計
                    self.episode_steps_history = []
                    self.episode_loss_history = []
                    self.episode_return_history = []
                
                return self.reset()
        
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
            r_ext_bar = self.ext_reward_sum / self.c_steps
            self.seg_buffer[-(self.args.n_agents+1)]["reward"] = r_ext_bar
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
            self.ext_reward_sum = 0.0
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
            self.prev_goal_vec    = self.goal_t.squeeze().cpu().numpy()
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
            self.prev_goal_vec = self.goal_t.squeeze().cpu().numpy()
            self.prev_global_state = global_state.copy()

        self.step_cnt += 1

        # -------- Worker 前傳 --------
        wk_tensor = torch.tensor(np.asarray(local_states), dtype=torch.float32,
                               device=self.device).view(1,1,self.args.n_agents,-1)  # [1,1,5,47]
        
        with torch.no_grad():
            logits, val_avg_wk, (val_w1, val_w2), self.worker_hidden = \
                self.worker(wk_tensor, self.worker_hidden, self.goal_t)
        
        # 檢查是否有敵人
        has_enemy = len(features.contacts[self.player_side]) > 0

        # logits shape: [T=1, B=1, A, n_actions]
        A = self.args.n_agents
        n_actions = self.args.n_actions
        # ---------- 建立 action mask（目前全部允許） ----------
        masks = []
        for i in range(A):                      # A = self.args.n_agents
            mask = torch.ones(n_actions, dtype=torch.bool, device=self.device)   # 先全部允許
            # 目前動作 0‥6 都是「指派任務」，沒有攻擊行為，
            # 不用再依偵測敵人／彈藥量去關閉任何按鍵
            masks.append(mask)
        # -----------------------------------------------------------

        # 對每個 agent 采樣動作
        actions = []
        # 建立 action mask、覆蓋 logits
        logits_masked = logits.clone()           # [1,1,A,4] 
        for j in range(A):
            logits_masked[0,0,j][~masks[j]] = -1e9   # 等同 -∞
        
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
        self.prev_state = local_states
        self.prev_action = actions
        
        # -- 每個 step 結束前，把「這一步」存成下一步的 baseline --
        self.AOE_last_states = deepcopy(self.AOE_states)

        # 更新友軍存活狀態並分配動作
        self.friendly_info.update_alive(features)
        action_cmd = ""
        for idx, name in enumerate(self.friendly_info.order):
            if not self.friendly_info.alive[name]:
                continue
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            # rule-based stop if reached goal
            if local_states[idx][0] < self.done_condition:
                action_cmd += "\n" + set_unit_heading_and_speed(
                    side=self.player_side,
                    unit_name=name,
                    heading=unit.CH,
                    speed=0
                )
            else:
                action_cmd += "\n" + self.apply_action(actions[idx], unit, features)

        if self.episode_step < 10:
            for unit in features.units[self.enemy_side]:
                action_cmd += "\n" + set_unit_to_mission(
                    unit_name=unit.Name,
                    mission_name='C_PatrolSCS1'
                )
            self.logger.debug(f"應用動作命令: {action_cmd}")
        
        # 更新前一步資料
        self.prev_state = local_states
        self.prev_action = actions
        self.total_steps += 1
        self.episode_step += 1
        self.episode_reward += rewards[0]


        
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

        # 友軍
        for name in self.friendly_info.order[:self.args.n_agents]:
            u = self.get_unit_info_from_observation(features, self.player_side, name)
            if u is None:
                vec.extend([0,0,0,0,0]); continue

            dist_norm, s, c = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH,
                                                  features, u.Name)
            mount_ratio = self._calc_mount_ratio(u)
            vec.extend([1, dist_norm, s, c, mount_ratio])

        # 敵軍
        for name in self.enemy_info.order[:self.args.enemy_num]:
            u = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if u is None:
                vec.extend([0,0,0,0]); continue

            dist_norm, s, c = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH,
                                                  features, u.Name)
            vec.extend([1, dist_norm, s, c])

        vec = np.asarray(vec, dtype=np.float32)
        assert len(vec) == self.GLOBAL_DIM, f"Global state dimension mismatch: {len(vec)} != {self.GLOBAL_DIM}"
        return vec

    def _get_unit_ll(self, features, side, name):
        u = self.get_unit_info_from_observation(features, side, name)
        return (float(u.Lon), float(u.Lat)) if u else (None, None)

    def _rel_to_target(self, lon, lat, ch_deg, features, unit_name):
        """
        回傳 (dist_norm, sinθ, cosθ)；依『這艘船』決定它要朝哪裡：
          - AOE 本身：直奔台中港
          - 其餘護航艦：追隨 AOE 當前位置
        """
        # 1) 決定目標 Lon/Lat
        if unit_name in self.AOE_list:              # AOE → 台中港
            tgt_lon, tgt_lat = self.TC_PORT_LON, self.TC_PORT_LAT
        else:                                       # 護航艦 → AOE
            tgt_lon, tgt_lat = self._get_unit_ll(features, self.player_side, self.AOE_list[0])
            if tgt_lon is None:                     # AOE 已沉：退而求其次 → 港口
                tgt_lon, tgt_lat = self.TC_PORT_LON, self.TC_PORT_LAT

        # 2) 距離與角度（完全沿用舊演算法）
        earth_r   = 6371.0
        lon_scale = np.cos(np.radians(lat))
        dx = (tgt_lon - lon) * np.pi * earth_r * lon_scale / 180.0
        dy = (tgt_lat - lat) * np.pi * earth_r / 180.0
        dist_norm = np.sqrt(dx*dx + dy*dy) / self.max_distance

        target_ang = np.arctan2(dy, dx)
        head_rad   = np.deg2rad(90.0 - ch_deg)
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
        for name in self.friendly_info.order[:self.args.n_agents]:
            u = self.get_unit_info_from_observation(features, self.player_side, name)
            if u is None:     # 已沉沒
                continue
            dist_norm, _, _ = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH,
                                                  features, u.Name)
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
        dist_norm, rel_sin, rel_cos = self._rel_to_target(ac_lon, ac_lat, ac.CH,
                                                          features, ac.Name)
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

        # ----------- 友軍資訊 (固定 A-1 艘) -------------
        friend_states = []

        # 1. 取出前 A 艘友軍名單
        friend_names = self.friendly_info.order[:self.args.n_agents]

        # 2. 去掉自己
        friend_other = [n for n in friend_names if n != ac.Name]

        # 3. 強制長度 = A-1
        need = self.args.n_agents - 1
        if len(friend_other) < need:
            friend_other += [None] * (need - len(friend_other))   # 不足補 None
        else:
            friend_other = friend_other[:need]                    # 過長就裁掉

        # 4. 組裝向量
        for fname in friend_other:
            if fname is None:                 # 用 0 占位
                friend_states.extend([0, 0, 0, 0, 0])
                continue

            u = self.get_unit_info_from_observation(
                features, self.player_side, fname)

            if u is None:                     # 死亡
                friend_states.extend([0, 0, 0, 0, 0])
                continue

            dist_norm, rel_sin, rel_cos = self._rel_to_target(
                float(u.Lon), float(u.Lat), u.CH, features, u.Name)
            mount_ratio = self._calc_mount_ratio(u)
            friend_states.extend([1, dist_norm, rel_sin, rel_cos, mount_ratio])

        # 敵軍資訊
        enemy_states = []
        for name in self.enemy_info.order[:self.args.enemy_num]:
            u = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if u is None:
                enemy_states.extend([0, 0, 0, 0])
                continue
            
            dist_norm, rel_sin, rel_cos = self._rel_to_target(float(u.Lon), float(u.Lat), u.CH,
                                                              features, u.Name)
            enemy_states.extend([1, dist_norm, rel_sin, rel_cos])

        # 組合所有狀態
        state = np.concatenate([
            base_state,
            np.array(friend_states, dtype=np.float32),
            np.array(enemy_states, dtype=np.float32)
        ])
        assert len(state) == self.LOCAL_DIM, f"Local state dimension mismatch: {len(state)} != {self.LOCAL_DIM}"
        return state
    
    def get_states(self, features: Multi_Side_FeaturesFromSteam) -> list[np.ndarray]:
        """
        獲取所有友方單位的狀態向量列表
        
        參數:
            features: 當前環境的觀察資料
            
        返回:
            所有友方單位的狀態向量列表
        """
        states = []
        # 對每個初始友軍單位按順序生成狀態，死亡的單位回傳默認值
        for name in self.friendly_info.order[:self.args.n_agents]:
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:
                # 單位死亡或不存在，返回預設零state
                state = np.zeros(self.LOCAL_DIM, dtype=np.float32)
                # state = self.normalize_state(raw_state)
                # print(f"單位 {name} 死亡或不存在，返回預設零state: {state}")
            else:
                state = self.get_state(features, unit)
            states.append(state)
        return states
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    # ------------------------------------------------------------
    # 取代舊版 get_reward
    # ------------------------------------------------------------
    def get_reward(self,
                   state: np.ndarray,
                   next_state: np.ndarray,
                   is_win: bool,
                   is_fail: bool) -> float:
        """
        回傳「未縮放」的 raw_reward
          • AOE 前進 1 km ≈ +100
          • 殲滅 1 艘敵艦 +10
          • 勝利 +self.win_reward，失敗 +self.fail_reward
        """
        raw_reward = 0.0

        # -------- AOE 推進獎勵 -----------------------------------
        if not is_fail:                                 # AOE 沒沉才能計
            for aoe in self.AOE_list:
                if aoe in self.AOE_last_states and aoe in self.AOE_states:
                    # state[0] = dist_norm；用「距離變小」給分
                    raw_reward += 100 * \
                        (self.AOE_last_states[aoe][0] - self.AOE_states[aoe][0])

        # -------- 殲敵獎勵 ---------------------------------------
        if self.enemy_info.enemy_alive_count < self.enemy_info.prev_enemy_alive_count:
            diff = (self.enemy_info.prev_enemy_alive_count -
                    self.enemy_info.enemy_alive_count)
            raw_reward += 100 * diff
        # 更新基準
        self.enemy_info.prev_enemy_alive_count = self.enemy_info.enemy_alive_count

        # -------- 勝負 -------------------------------------------
        if is_win:
            raw_reward += self.win_reward
        if is_fail:
            raw_reward += self.fail_reward

        return raw_reward
    
    # ------------------------------------------------------------
    # 取代舊版 get_rewards
    # ------------------------------------------------------------
    def get_rewards(self,
                    features: Multi_Side_FeaturesFromSteam,
                    state: list[np.ndarray],
                    next_state: list[np.ndarray]) -> list[np.ndarray]:
        """
        回傳「已縮放」的 rewards，AOE 不列入 worker → 只給護航艦
        """
        rewards: list[float] = []

        is_win  = self.get_win(features, next_state)
        is_fail = self.get_fail(features, next_state)

        # 1) 整理「護航艦序列」（排除 AOE） ------------------------
        order_wo_aoe = [n for n in self.friendly_info.order
                        if n not in self.AOE_list]

        # 2) 依序計算每艘護航艦的 reward ----------------------------
        for i, name in enumerate(order_wo_aoe):
            unit = self.get_unit_info_from_observation(
                features, self.player_side, name)

            if unit is None:                       # 護航艦陣亡
                reward = -10.0
            else:
                reward = self.get_reward(state[i], next_state[i],
                                          is_win=is_win, is_fail=is_fail)

            # -------- 縮放到 [-reward_scale, +reward_scale] -------
            scaled = reward / (self.max_return / self.reward_scale)
            rewards.append(scaled)

        return rewards

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
        missions = [
            'T_PatrolSCS1','T_PatrolSCS2','T_PatrolSCS3',
            'T_PatrolSCS4','T_PatrolSCS5','T_PatrolSCS6','T_PatrolSCS7'
        ]                                   # ⬅︎ 共 7 條

        if 0 <= action < len(missions):     # 0‥6 → 指派對應任務
            return actlib.set_unit_to_mission(
                unit_name   = ac.Name,
                mission_name= missions[action]
            )

        #   **再也不會有 action == 7**
        self.logger.warning(f"[apply_action] 非法動作 {action}")
        return ""

    def reassign_mission(self, features: Multi_Side_FeaturesFromSteam) -> str:
        """
        只負責產生指派命令，不負責決定什麼時候下達
        """
        cmd_lines = []

        # AOE 一律指派去『T_ToTaichung』
        for aoe in self.AOE_list:
            cmd_lines.append(
                actlib.set_unit_to_mission(
                    unit_name=aoe,
                    mission_name="T_ToTaichung",
                )
            )

        # 所有敵艦一律指派去『C_PatrolSCS1』
        for enemy in features.units[self.enemy_side]:
            cmd_lines.append(
                actlib.set_unit_to_mission(
                    unit_name=enemy.Name,
                    mission_name="C_PatrolSCS1",
                )
            )

        return "\n".join(cmd_lines)
    
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

        device = self.device
        Ag = self.args.n_agents + 1
        full_len = (len(traj) // Ag) * Ag     # 可整除的部分
        traj = traj[:full_len]                # 丟掉最後不足 Ag 的零頭
        T = len(traj) // Ag

        # ---------- 1. 轉 list ----------
        state_g = []; state_l = []; goal = []
        action  = []; logp = []; value = []; reward = []
        done = []; cost = []
        for t in traj:
            state_g.append(t["state_g"] if t["state_g"] is not None else np.zeros(self.GLOBAL_DIM))
            state_l.append(t["state_l"] if t["state_l"] is not None else np.zeros(self.LOCAL_DIM))
            goal.append(t["goal"])
            action.append(-1 if t["action"] is None else t["action"])   # manager=-1
            logp.append(t["logp"]); value.append(t["value"]); reward.append(t["reward"])
            done.append(t["done"]); cost.append(t["cost"])

        # --------------- 2. 重新組 batch --------------- #
        A  = self.args.n_agents          # =5
        Ag = A + 1                       # =6   (含 manager)

        # 先把 python list → numpy → tensor
        state_g_raw = torch.from_numpy(np.asarray(state_g , np.float32)).to(device).view(T,Ag,-1)
        state_l_raw = torch.from_numpy(np.asarray(state_l , np.float32)).to(device).view(T,Ag,-1)
        goal_raw    = torch.from_numpy(np.asarray(goal    , np.float32)).to(device).view(T,Ag,-1)
        action_raw  = torch.from_numpy(np.asarray(action  , np.int64 )).to(device).view(T,Ag)
        reward_raw  = torch.from_numpy(np.asarray(reward  , np.float32)).to(device).view(T,Ag)
        done_raw    = torch.from_numpy(np.asarray(done    , np.float32)).to(device).view(T,Ag)            # 1=終結
        cost_raw    = torch.from_numpy(np.asarray(cost    , np.float32)).to(device).view(T,Ag)            # Manager 部分才有值

        # ---- Manager ----
        mgr_state   = state_g_raw[:,0,:]                   # [T,45]
        mgr_in      = mgr_state[:,None,:]   # [T,1,45]
        h0m         = self.manager.init_hidden(batch_size=1)      # (1,1,64)

        # ---- Worker ----
        wk_state    = state_l_raw[:,1:,:]                  # [T,A,47]  (去掉 manager 的佔位)
        wk_in       = wk_state[:,None,:,:]                # [T,1,A,47]
        goal_in     = goal_raw[:,1:,:][:,None,:,:]        # [T,1,A,goal_dim]
        h0w         = self.worker.init_hidden(batch_size=1)       # (1, 5, 64)

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
        entropy = dist.entropy().mean()

        loss_act_wk = -(adv_wk.detach() * logp_wk).mean()

        # --- Manager value loss (two heads) ---
        val_mgr1 = val_mgr1.squeeze(1)      # [T]
        val_mgr2 = val_mgr2.squeeze(1)
        
        # 計算變異數比例自動對齊
        var_wk  = adv_wk.pow(2).mean().detach()
        var_mgr = adv_mgr.pow(2).mean().detach()
        mgr_scale = torch.sqrt((var_wk + 1e-8) / (var_mgr + 1e-8))
        
        loss_val_mgr = 0.25 * (
            (mgr_scale * (return_mgr - val_mgr1)).pow(2).mean() +
            (mgr_scale * (return_mgr - val_mgr2)).pow(2).mean()
        )

        # --- Worker value loss (two heads) ---
        val_wk1 = val_wk1.squeeze(1).squeeze(-1)   # [T, A]
        val_wk2 = val_wk2.squeeze(1).squeeze(-1)
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
        
        self.logger.info(f"total_loss={total_loss.item():.6f}")

        self.manager_optimizer.zero_grad()
        self.worker_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 5)
        self.manager_optimizer.step()
        self.worker_optimizer.step()
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
        action_cmd = self.reassign_mission(self.last_features) if hasattr(self, "last_features") else ""
        action_cmd += self.reset_cmd
        
        return action_cmd
    
    def get_reset_cmd(self, features: Multi_Side_FeaturesFromSteam):
        """
        產生重置遊戲所需的命令
        
        參數:
            features: 當前環境的觀察資料
        """
        action_cmd = ""
        for ac in features.units[self.player_side]:
            action_cmd += delete_unit(
                side=self.player_side,
                unit_name=ac.Name
            ) + "\n"
            action_cmd += add_unit(
                type='Ship',
                unitname=ac.Name,
                dbid=ac.DBID,
                side=self.player_side,
                Lat=ac.Lat,
                Lon=ac.Lon
            ) + "\n"
        for ac in features.units[self.enemy_side]:
            action_cmd += delete_unit(
                side=self.enemy_side,
                unit_name=ac.Name
            ) + "\n"
            action_cmd += add_unit(
                type='Ship',
                unitname=ac.Name,
                dbid=ac.DBID,
                side=self.enemy_side,
                Lat=ac.Lat,
                Lon=ac.Lon
            ) + "\n"
            action_cmd += set_unit_to_mission(
                unit_name=ac.Name,
                mission_name='C_PatrolSCS1'
            ) + "\n"
        self.reset_cmd = action_cmd