from pycmo.lib.actions import (
    set_unit_position,
    set_unit_heading_and_speed,
    delete_unit,          # ★新增
    add_unit,             # ★新增
    set_unit_to_mission,   # ← 這樣就夠
    AvailableFunctions
)
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
import numpy as np
from collections import deque

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import logging
import math
from typing import Tuple
import time

# 导入FeUdal模型
from scripts.FeUdal11.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic

class MyAgent(BaseAgent):
    def __init__(self, player_side: str, ac_name: str, target_name: str = None, destination: dict = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param ac_name: 控制的單位名稱（例如 B 船）
        :param target_name: 目標單位名稱（例如 A 船），可選
        :param destination: 目的地經緯度字典，可選
        :param log_level: 日誌級別，預設為INFO，可設置為logging.DEBUG啟用詳細日誌
        """
        super().__init__(player_side)
        self.ac_name = ac_name
        self.target_name = target_name  # 用於追蹤特定目標（例如 A 船）
        # 如果外部沒指定 destination，就用金門 (Kinmen) 的固定經緯度
        self.destination = destination if destination is not None else {
            'Lon': 118.22,
            'Lat': 24.27
        }
        
        self.reset_cmd = ""          # 存放下一回合要執行的 Lua 指令
        
        # --- 新增：存初始狀態 ---
        self.init_snapshot = {
            'friendly': {},   # {unit_name: {'Lat':..., 'Lon':..., 'DBID':..., 'Mission':...}}
            'enemy':    {}
        }
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(f"MyAgent_{ac_name}")
        self.logger.setLevel(logging.INFO)
        
        # ← 新增這段 ↓
        fh = logging.FileHandler(f"logs/{self.ac_name}.log", mode="a")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        # ← 新增結束
        
        # FeUdal网络参数
        class Args:
            def __init__(self):
                self.manager_hidden_dim = 64
                self.worker_hidden_dim = 64
                self.state_dim_d = 5
                self.embedding_dim_k = 16
                self.n_actions = 5
                self.manager_dilation = 10  # 改為 c=10

        self.args = Args()
        self.input_size = 5  # [rel_x, rel_y, sinθ, cosθ, enemy_exists]
        
        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # 初始化FeUdal网络
        self.manager = Feudal_ManagerAgent(self.input_size, self.args).to(self.device)
        self.worker = Feudal_WorkerAgent(self.input_size, self.args).to(self.device)
        
        # —— 改為雙 Critic
        self.worker_critic = FeUdalCritic(self.input_size, self.args).to(self.device)
        self.manager_critic = FeUdalCritic(self.input_size, self.args).to(self.device)
        
        # Worker + its critic
        self.worker_optimizer = torch.optim.Adam(
            list(self.worker.parameters()) + list(self.worker_critic.parameters()),
            lr=5e-4
        )
        # Manager + its critic
        self.manager_optimizer = torch.optim.Adam(
            list(self.manager.parameters()) + list(self.manager_critic.parameters()),
            lr=5e-4
        )
        
        # On-policy 和 GAE 相關設定
        self.episode_memory = []      # list of (state, action, reward, next_state, done)
        self.gae_lambda = 0.95        # GAE 的 λ
        self.gamma = 0.99             # 折扣因子
        self.total_steps = 0
        self.done_condition = 0.2     # 歸一化距離閾值（約 0.05 ~ 0.1）
        self.entropy_coef = 0.02      # 提高探索熵，鼓勵更多嘗試

        # 初始化隐藏状态
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()

        self.best_distance = 2.0      # 初始設為最大可能距離（√2）√2≈1.414→2.0
        self.worst_distance = 0.0     # 最差距離（初始為0）
        self.total_reward = 0

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None
        self.prev_action = None
        self.prev_raw_goal = None

        # 初始化訓練統計記錄器（如果有的話）
        self.stats_logger = Logger()
        
        # 添加遊戲結束標記
        self.episode_step = 0
        self.max_episode_steps = 1000
        self.episode_done = False
        self.episode_reward = 0
        
        # 添加步數計數器
        self.step_counter = 0

        # 在第一回合取到 unit 時填好：
        self.ship_type = None
        self.ship_dbid = None
        # 你原本 reset() 用的座標
        self.init_lat, self.init_lon = 23.38, 119.19

        # ---------- 新增开始 ----------
        # 敵方艦艇初始屬性（第一次 record 用）
        self.enemy_type = None
        self.enemy_dbid = None
        self.enemy_side = None
        # 敵方 spawn 座標，和你的 config 保持一致
        self.enemy_init_lat, self.enemy_init_lon = 24.01, 118.20
        # ---------- 新增结束 ----------

    def get_unit_info_from_observation(self, features: FeaturesFromSteam, unit_name: str) -> Unit:
        """
        從觀察中獲取指定單位的資訊。
        """
        units = features.units
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None
    
    def get_contact_info_from_observation(self, features: FeaturesFromSteam, contact_name: str) -> dict:
        """
        從觀察中獲取指定接觸點（敵方單位）的資訊。
        """
        contacts = features.contacts
        for contact in contacts:
            if contact['Name'] == contact_name:
                return contact
        return None
    
    def get_norm_distance(self, state: np.ndarray) -> float:
        """
        直接在歸一化後的 5 維空間裡 [rel_x, rel_y, sinθ, cosθ, exists]
        計算到目標的距離。
        """
        # state = [rel_x, rel_y, sinθ, cosθ, exists]
        rel_x, rel_y = state[0], state[1]
        return np.sqrt(rel_x**2 + rel_y**2)

    def get_extrinsic_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        reward = 0

        # 距離計算（真實距離）
        distance = self.get_norm_distance(state)
        next_distance = self.get_norm_distance(next_state)

        # 距離差獎勵（係數從 100 降到 10）
        reward += 100 * (distance - next_distance)

        # 記錄距離變化到日誌
        self.logger.debug(f"距離變化: {distance:.4f} -> {next_distance:.4f}")

        # 新的最佳距離（距離明顯變近）
        if next_distance + 0.01 < self.best_distance:
            self.best_distance = next_distance
            reward += 1
            self.logger.debug(f"新的最佳距離: {self.best_distance:.4f}")

        # 新的最差距離（明顯變遠）
        if next_distance - 0.01 > self.worst_distance:
            self.worst_distance = next_distance
            reward -= 1
            self.logger.debug(f"新的最差距離: {self.worst_distance:.4f}")

        # —— 加這段：分段式漸進獎勵，從距離 0.6 開始 —— #
        if next_distance < 0.2:
            reward += 5
        elif next_distance < 0.3:
            reward += 4
        elif next_distance < 0.4:
            reward += 3
        elif next_distance < 0.5:
            reward += 2
        elif next_distance < 0.6:
            reward += 1

        # 抵達目標的高額獎勵
        if next_distance < self.done_condition:
            reward += 200

        # 超出地圖邊界懲罰
        if abs(next_state[0]) > 1:
            reward -= 15
        if abs(next_state[1]) > 1:
            reward -= 15

        return float(reward)

    def compute_intrinsic_rewards(self, states, raw_goals, c):
        """
        計算內在獎勵（基於目標和狀態變化的相似度）
        """
        intrinsic_rewards = []
        for t in range(len(raw_goals)):
            if t + c < len(states):
                delta = states[t+c] - states[t]
                # 投影 delta 到與 raw_goal 相同的維度
                delta_emb = self.manager.delta_fc(delta)
                cos_sim = F.cosine_similarity(
                    F.normalize(delta_emb, dim=-1, p=2),
                    F.normalize(raw_goals[t], dim=-1, p=2),
                    dim=-1
                ).clamp(-1, 1)
                intrinsic_rewards.append(cos_sim)
        if len(intrinsic_rewards) == 0:
            return torch.zeros(0, device=states.device, dtype=torch.float32)
        return torch.stack(intrinsic_rewards).squeeze(-1)  # 確保輸出是 [T]

    def action(self, features: FeaturesFromSteam, VALID_FUNCTIONS: AvailableFunctions) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        # 列印動態偵測到的 contacts（真正從 CMO 拿到的清單）
        contacts = features.contacts or []
        # print(f"偵測到 {len(contacts)} 個 Contacts:")
        for c in contacts:
            # c 是一個 dict，裡面通常有 ID, Name, CS, CA, Lon, Lat 等欄位
            print(f"  → ID={c.get('ID')}, Name={c.get('Name')}, Lon={c.get('Lon')}, Lat={c.get('Lat')}")
        
        # 我方／敵方列印
        print("我方軍艦:", [u.Name for u in features.units])
        all_sides = features.get_sides()
        enemy_side = next(s for s in all_sides if s != self.player_side)
        enemy_units = features.get_side_units(enemy_side)
        print(f"敵方軍艦({enemy_side}):", [u.Name for u in enemy_units])

        # 記錄初始快照（如果還沒記錄）
        if not self.init_snapshot['friendly']:          # 只做一次
            # 收 friendly
            for u in features.units:
                if u.Side == self.player_side:
                    self.init_snapshot['friendly'][u.Name] = {
                        'Lat': float(u.Lat),
                        'Lon': float(u.Lon),
                        'DBID': u.DBID,
                        'Mission': getattr(u, 'AssignedMission', None)
                    }
            # 收 enemy（此時若偵測不到先留空，之後再補）
            for u in enemy_units:
                self.init_snapshot['enemy'][u.Name] = {
                    'Lat': float(u.Lat),
                    'Lon': float(u.Lon),
                    'DBID': u.DBID,
                    'Mission': getattr(u, 'AssignedMission', None)
                }
            # 若 enemy 仍空，下一個 observation 再補
            if not self.init_snapshot['enemy'] and enemy_units:
                for u in enemy_units:
                    self.init_snapshot['enemy'][u.Name] = {
                        'Lat': float(u.Lat),
                        'Lon': float(u.Lon),
                        'DBID': u.DBID,
                        'Mission': getattr(u, 'AssignedMission', None)
                    }
            self.logger.info("已記錄初始艦艇狀態快照")
        
        # 記錄敵方陣營名稱
        if self.enemy_side is None:
            self.enemy_side = enemy_side
        
        print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        
        # 如果找不到單位，可能是被擊沉或其他原因
        if ac is None:
            # 確保我們已經記錄過船隻類型和 ID（至少有初始值）
            if self.ship_type is None or self.ship_dbid is None:
                self.logger.error(f"單位消失且尚未記錄船隻類型和 ID，無法重生")
                return ""  # 返回空命令
            
            self.logger.warning(f"找不到單位: {self.ac_name}，可能已被擊沉，執行重生")
            # ★ 先組好指令
            self.build_reset_cmd()
            return self.reset()
            
        # 立即記錄船隻資訊（無論是首次還是更新）
        if hasattr(ac, 'DBID') and hasattr(ac, 'Type'):
            if self.ship_dbid is None:
                self.ship_dbid = ac.DBID
                self.ship_type = ac.Type
                self.logger.info(f"首次記錄艦艇資訊：Type={self.ship_type}, DBID={self.ship_dbid}")
            elif self.ship_dbid != ac.DBID:
                # 如果 DBID 變化（可能是因為重生後 CMO 賦予了新 ID）
                self.logger.info(f"艦艇 DBID 已更新：舊={self.ship_dbid}，新={ac.DBID}")
                self.ship_dbid = ac.DBID
        
        # 取真正的敵方 Unit
        enemy_unit = next((u for u in enemy_units if u.Name == self.target_name), None)
        if enemy_unit:
            if self.enemy_dbid is None:
                self.enemy_dbid = enemy_unit.DBID
                self.enemy_type = enemy_unit.Type
                self.logger.info(f"首次記錄敵方艦艇：Type={self.enemy_type}, DBID={self.enemy_dbid}")
            elif self.enemy_dbid != enemy_unit.DBID:
                # 如果是重生後 CMO 給了新 DBID，也更新
                self.logger.info(f"敵方艦艇 DBID 更新：{self.enemy_dbid} → {enemy_unit.DBID}")
                self.enemy_dbid = enemy_unit.DBID
        
        # 找到我們要追蹤的敵方 contact
        contact = self.get_contact_info_from_observation(features, self.target_name)
        print("DEBUG contact:", contact)
        
        # 獲取當前狀態
        current_state = self.get_state(features)
        
        # 如果有前一步資料，收集經驗
        if self.prev_state is not None and self.prev_action is not None:
            ext_reward = self.get_extrinsic_reward(self.prev_state, current_state)
            done = self.get_norm_distance(current_state) < self.done_condition
            self.total_reward += ext_reward
            self.episode_reward += ext_reward
            
            # 將經驗存儲到 episode_memory
            self.episode_memory.append((
                self.prev_state,      # state_t
                self.prev_action,     # action_t
                ext_reward,           # extrinsic reward
                current_state,        # state_{t+1}
                done,                 # done flag
                self.prev_raw_goal    # raw goal for intrinsic reward
            ))
            
            # 如果 episode 結束，就更新，然後 clear episode_memory
            if done or self.episode_step >= self.max_episode_steps:
                self.episode_done = True
                self.logger.info(f"遊戲結束! 總獎勵: {self.episode_reward:.4f}")
                self.train_on_episode()
                self.episode_memory.clear()

                # ★ 先組好指令
                self.build_reset_cmd()

                return self.reset()
        
        # 選擇動作 - 只進行一次 forward，同時獲取用於取樣和保留梯度的目標
        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Manager forward 一次，獲取原始 goal 和 raw_goal (保留梯度)
        goal_raw, (h_m, c_m), raw_goal_raw = self.manager(state_tensor, self.manager_hidden)
        
        # 更新 hidden state，但使用 detach 避免跨時間步梯度爆炸
        self.manager_hidden = (h_m.detach(), c_m.detach())
        
        # 對 Worker 使用 detached 版本的目標
        goal_sample = goal_raw.detach()
        
        # Worker 使用 detached 目標選擇動作
        logits, (h_w, c_w) = self.worker(
            state_tensor, 
            self.worker_hidden,
            goal_sample
        )
        
        # 新增：Worker hidden state 也需要 detach，避免跨時間步梯度積累
        self.worker_hidden = (h_w.detach(), c_w.detach())
        
        # 使用 Categorical 分布采样动作
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        
        # 執行動作
        action_cmd = self.apply_action(action, ac, contacts, VALID_FUNCTIONS)
        
        # 更新前一步資料 - 存保留梯度的 raw_goal_raw
        self.prev_state = current_state
        self.prev_action = action
        self.prev_raw_goal = raw_goal_raw  # 下一步做經驗存取時使用可 backprop 的那個
        self.total_steps += 1
        self.episode_step += 1

        # 每一步都確保敵方單位在巡邏任務中
        for unit in features.get_side_units(self.enemy_side):
            action_cmd += "\n" + set_unit_to_mission(
                unit.Name,          # unitname
                "Kinmen patrol"     # ← 跟 Mission Editor 內完全一致
            )

        return action_cmd
    
    def normalize_state(self, state):
        min_lon, max_lon = 118.2, 119.2  # 根據你的地圖範圍調整
        min_lat, max_lat = 23.3, 24.3
        min_heading, max_heading = 0.0, 360.0
        min_speed, max_speed = 0.0, 35.0
        norm_state = np.zeros_like(state)
        norm_state[0] = (state[0] - min_lon) / (max_lon - min_lon)  # B_lon
        norm_state[1] = (state[1] - min_lat) / (max_lat - min_lat)  # B_lat
        norm_state[2] = state[2] / max_heading  # B_heading
        norm_state[3] = state[3] / max_speed  # B_speed   
        norm_state[4] = (state[4] - min_lon) / (max_lon - min_lon)  # A_lon
        norm_state[5] = (state[5] - min_lat) / (max_lat - min_lat)  # A_lat
        return norm_state   

    def get_state(self, features: FeaturesFromSteam) -> np.ndarray:
        """
        回傳新的 5 維狀態：[rel_x, rel_y, sinθ, cosθ, enemy_exists]
        """
        # 1) 拿我方船經緯度
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            return np.zeros(5, dtype=np.float32)

        lon_B, lat_B = float(ac.Lon), float(ac.Lat)

        # 2) 拿金門目的地經緯度
        lon_D = float(self.destination['Lon'])
        lat_D = float(self.destination['Lat'])

        # 3) normalization 範圍，跟 normalize_state 保持一致
        min_lon, max_lon = 118.2, 119.2
        min_lat, max_lat = 23.3, 24.3

        # 4) normalized positions
        norm_B_lon = (lon_B - min_lon) / (max_lon - min_lon)
        norm_B_lat = (lat_B - min_lat) / (max_lat - min_lat)
        norm_D_lon = (lon_D - min_lon) / (max_lon - min_lon)
        norm_D_lat = (lat_D - min_lat) / (max_lat - min_lat)

        # 5) 相對座標差值
        rel_x = norm_D_lon - norm_B_lon
        rel_y = norm_D_lat - norm_B_lat

        # 6) 方位角 θ，atan2 是 (Δy, Δx)
        θ = math.atan2(rel_y, rel_x)
        sinθ = math.sin(θ)
        cosθ = math.cos(θ)

        # 7) 偵測敵艦是否存在
        exists = 0.0
        if self.target_name:
            contact = self.get_contact_info_from_observation(features, self.target_name)\
                      or self.get_unit_info_from_observation(features, self.target_name)
            if contact:
                exists = 1.0

        # 8) 組成 state 並回傳
        state = np.array([rel_x, rel_y, sinθ, cosθ, exists], dtype=np.float32)
        return state

    def apply_action(self,
                     action: int,
                     ac: Unit,
                     contacts: list,
                     vf: list) -> str:
        """
        action: 0-3 是方向移動，4 是攻擊。
        contacts: CMO 返回的所有 contact dict 列表
        """
        # 1) 先處理 0-3 四個移動命令
        if action == 0:   # 北
            return set_unit_heading_and_speed(self.player_side, self.ac_name, 0, 30)
        elif action == 1: # 東
            return set_unit_heading_and_speed(self.player_side, self.ac_name, 90, 30)
        elif action == 2: # 南
            return set_unit_heading_and_speed(self.player_side, self.ac_name, 180, 30)
        elif action == 3: # 西
            return set_unit_heading_and_speed(self.player_side, self.ac_name, 270, 30)

        # 2) action == 4 → 只有真的偵測到目標時才攻擊
        if action == 4:
            target = None
            for c in contacts:
                # 用 CS 或 Name 判斷是否為可攻擊的 FFG 目標
                if c.get('CS') == 'FFG' or c.get('Name', '').startswith('FFG'):
                    target = c
                    break

            if target and 'ID' in target:
                fn_list = [f for f in vf if f.name == "auto_attack_contact"]
                if fn_list:
                    fn = fn_list[0]
                    return fn.corresponding_def(ac.ID, target['ID'])
            # 如果 contacts 裡找不到敵人，就不動作
            return ""

        # 3) 其餘所有情況（action 不在 0-4）都回空
        return ""

    def train_on_episode(self):
        """使用當前 episode 的數據進行 on-policy 訓練"""
        # 防止空 memory 的 guard，如果 memory 空就直接 return
        if len(self.episode_memory) == 0:
            return

        # 從 episode_memory 拆欄
        states, actions, ext_rewards, next_states, dones, raw_goals = zip(*self.episode_memory)
        T = len(ext_rewards)
        # 這裡用 dilation 代表 Manager 的目標展開長度
        dilation = self.args.manager_dilation

        # 檢查資料長度
        if T <= dilation:     # episode 太短，資料不足以對齊 c-step
            self.logger.warning(
                f"Episode length {T} ≤ dilation {dilation}; skip training.")
            return

        # 轉 tensor
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        ext_rewards = torch.tensor(ext_rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # 立刻把原始 raw_goals detach 掉，再重新開啟 requires_grad
        raw_goals_mgr = torch.stack(raw_goals).squeeze(1)      # Manager 用，保留梯度
        raw_goals_wrk = raw_goals_mgr.detach()                 # Worker/Intrinsic 用，斷梯度

        # --- Worker 部分 ---
        values_W = self.worker_critic(states).squeeze(-1)
        with torch.no_grad():
            next_values_W = self.worker_critic(next_states).squeeze(-1)

        # 計算內在獎勵並對齊切片，使用 raw_goals_wrk
        intrinsic_rewards = self.compute_intrinsic_rewards(states, raw_goals_wrk, dilation)

        # 把外在獎勵也加進來
        ext = ext_rewards[dilation:]       # 切片對齊後的 extrinsic reward
        lambda_ext = 0.8                # extrinsic 權重係數
        dr = 2 * intrinsic_rewards + lambda_ext * ext

        nv = next_values_W[dilation:]
        dv = values_W[dilation:]
        d = dones[dilation:]

        # GAE for Worker (基於內在獎勵)
        deltas_W = dr + self.gamma * nv * (1 - d) - dv
        advantages_W = torch.zeros_like(dr, device=self.device)
        last_adv_W = 0.0
        for t in reversed(range(len(dr))):
            last_adv_W = deltas_W[t] + self.gamma * self.gae_lambda * (1 - d[t]) * last_adv_W
            advantages_W[t] = last_adv_W

        # --- Manager 部分（改用外在獎勵） ---
        values_M = self.manager_critic(states).squeeze(-1)[dilation:]
        next_values_M = self.manager_critic(next_states).squeeze(-1)[dilation:]
        dones_M = dones[dilation:]

        # GAE for Manager (基於外在獎勵)
        er = ext_rewards[dilation:]
        deltas_M = er + self.gamma * next_values_M * (1 - dones_M) - values_M
        advantages_M = torch.zeros_like(er, device=self.device)
        last_adv_M = 0.0
        for t in reversed(range(len(er))):
            last_adv_M = deltas_M[t] + self.gamma * self.gae_lambda * (1 - dones_M[t]) * last_adv_M
            advantages_M[t] = last_adv_M

        # 正規化優勢
        advantages_W = (advantages_W - advantages_W.mean()) / (advantages_W.std() + 1e-8)
        advantages_M = (advantages_M - advantages_M.mean()) / (advantages_M.std() + 1e-8)

        # --- Worker: 逐步 unroll LSTM ---
        h_w, c_w = self.worker.init_hidden()
        all_log_probs = []
        all_entropy = []
        for t in range(T):
            state_t = states[t].unsqueeze(0)
            goal_t = raw_goals_wrk[t].unsqueeze(0)
            logits_t, (h_w, c_w) = self.worker(state_t, (h_w, c_w), goal_t)
            dist_t = Categorical(logits=logits_t)
            all_log_probs.append(dist_t.log_prob(actions[t].unsqueeze(0)))
            all_entropy.append(dist_t.entropy())

        # 修正 log_probs 的維度處理
        log_probs = torch.cat(all_log_probs).flatten()
        entropy = torch.stack(all_entropy).flatten()

        # Worker loss (對齊內在獎勵切片)
        worker_loss = -(advantages_W.detach() * log_probs[dilation:]).mean()
        returns_W = advantages_W.detach() + dv
        worker_value_loss = F.mse_loss(dv, returns_W)
        worker_entropy_loss = -self.entropy_coef * entropy[dilation:].mean()
        total_worker_loss = worker_loss + worker_value_loss + worker_entropy_loss

        # Worker 優化
        self.worker_optimizer.zero_grad()
        total_worker_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 1.0)
        self.worker_optimizer.step()

        # Manager 更新
        delta = states[dilation:] - states[:-dilation]        # c-step Δ
        delta_embs = self.manager.delta_fc(delta)
        cos_term = F.cosine_similarity(
            F.normalize(delta_embs, dim=-1, p=2),
            F.normalize(raw_goals_mgr[:-dilation], dim=-1, p=2),
            dim=-1
        ) + 1

        # Manager loss (使用外在獎勵的優勢)
        manager_policy_loss = -(advantages_M.detach() * cos_term).mean()
        returns_M = advantages_M.detach() + values_M
        manager_value_loss = F.mse_loss(values_M, returns_M)
        total_manager_loss = manager_policy_loss + manager_value_loss

        # Manager 優化
        self.manager_optimizer.zero_grad()
        total_manager_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 1.0)
        self.manager_optimizer.step()

        # 記錄訓練統計數據
        self.logger.info(f"Episode 訓練完成 - 步數: {self.total_steps}, "
                        f"Worker損失: {total_worker_loss.item():.4f}, "
                        f"Manager損失: {total_manager_loss.item():.4f}")
        
        # 記錄統計數據
        self.stats_logger.log_stat("total_worker_loss", total_worker_loss.item(), self.total_steps)
        self.stats_logger.log_stat(" total_manager_loss", total_manager_loss.item(), self.total_steps)
        self.stats_logger.log_stat("extrinsic_reward", er.mean().item(), self.total_steps)
        self.stats_logger.log_stat("intrinsic_reward", dr.mean().item(), self.total_steps)
        self.stats_logger.log_stat("return", self.episode_reward, self.total_steps)
        
        # —— 新增：記錄本回合結束時的最終歸一化距離 —— #
        final_state = next_states[-1].cpu().numpy()
        # 1) 先算出(可能是 numpy.float32)的距离
        final_norm_dist = self.get_norm_distance(final_state)
        # 2) 强制转成 Python 内建的 float，才能被 JSON 序列化
        final_norm_dist = float(final_norm_dist)

        # 3) 写入 stats_logger（这样它会出现在你的 JSON 里）
        self.stats_logger.log_stat("final_norm_dist", final_norm_dist, self.total_steps)
        # 4) 同步写入普通日志
        self.logger.info(f"Final normalized distance: {final_norm_dist:.4f}")
    
    def build_reset_cmd(self):
        """
        根據初始快照重建所有單位的位置和任務
        """
        # 檢查 friendly 快照
        if not self.init_snapshot['friendly']:
            self.logger.error("快照未準備好，無法 reset")
            self.reset_cmd = ""
            return

        # 檢查 enemy_side
        if self.enemy_side is None:
            self.logger.error("enemy_side 尚未設定，無法重置敵艦")
            self.enemy_side = "China"   # ← 或你劇本裡實際的紅方名稱

        # 檢查 enemy 快照
        if not self.init_snapshot['enemy']:
            self.logger.warning("敵艦快照尚未準備，先只重置友艦")

        cmd = []

        # ===== 我方：delete → add =====
        for name, info in self.init_snapshot['friendly'].items():
            cmd += [
                delete_unit(side=self.player_side, unit_name=name),  # 就算還活着，保險先刪
                add_unit(                                           # 重新生成
                    type='Ship',
                    unitname=name,
                    dbid=info['DBID'],
                    side=self.player_side,
                    Lat=info['Lat'],
                    Lon=info['Lon']
                )
            ]
            # 若開局時有任務就把它掛回去
            if info['Mission']:
                cmd.append(set_unit_to_mission(name, info['Mission']))

        # ===== 敵方：delete → add → assign =====
        for name, info in self.init_snapshot['enemy'].items():
            cmd += [
                delete_unit(side=self.enemy_side, unit_name=name),
                add_unit(type='Ship', unitname=name, dbid=info['DBID'],
                         side=self.enemy_side, Lat=info['Lat'], Lon=info['Lon'])
            ]
            # ★ 不再檢查 Mission 欄位，直接派回 PatrolTarget
            cmd.append(set_unit_to_mission(
                name,                 # unitname
                "Kinmen patrol"        # mission_name  ← 跟劇本裡名字完全一致
            ))

        self.reset_cmd = "\n".join(cmd)

    def reset(self) -> str:
        """
        清空內部狀態，並把剛剛 build 的 Lua 指令丟回 CMO
        """
        # -------- LSTM 狀態歸零 --------
        self.manager_hidden = tuple(h.detach() for h in self.manager.init_hidden())
        self.worker_hidden = tuple(h.detach() for h in self.worker.init_hidden())

        # -------- 內部統計歸零 --------
        self.best_distance = 2.0
        self.worst_distance = 0.0
        self.prev_state = None
        self.prev_action = None
        self.prev_raw_goal = None
        self.episode_step = 0
        self.episode_done = False
        self.episode_reward = 0
        self.episode_memory.clear()
        if hasattr(self.manager, "goal_buffer"):
            self.manager.goal_buffer.clear()

        # -------- 回傳指令 --------
        cmd = self.reset_cmd        # 先存起來
        self.reset_cmd = ""         # 清空以免下回重複
        return cmd