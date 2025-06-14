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
import torch.optim as optim

import logging
import math
from typing import Tuple
import time

# 导入FeUdal模型
from scripts.FeUdal11.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic

# -----------------------  常數區  -----------------------
ENEMY_MISSION = "5VS5"      # ★你的劇本裡真正的任務名稱
# -------------------------------------------------------

class MyAgent(BaseAgent):
    def __init__(self, player_side: str, manager_name: str, friendly_names: list = None, target_name: str = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param manager_name: 控制的單位名稱（例如 B 船）
        :param target_name: 目標單位名稱（例如 A 船），可選
        :param friendly_names: 友軍艦艇名稱列表，可選
        :param log_level: 日誌級別，預設為INFO，可設置為logging.DEBUG啟用詳細日誌
        """
        super().__init__(player_side)
        
        # --- Logger 先建 ---
        self.logger = logging.getLogger(f"MyAgent_{manager_name}")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f"logs/{manager_name}.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        self.logger.addHandler(fh)
        
        # 設定設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # ---- 放在最前面 ----
        self.manager_name = manager_name
        self.friendly_order = friendly_names or []   # 若空，首回合自動掃
        # --------------------
        
        # 設定目標
        self.target_name = target_name
        self.enemy_side = None
        self.enemy_dbid = None
        self.enemy_type = None
        
        # 設定船隻資訊
        self.ship_dbid = None
        self.ship_type = None
        
        # 設定距離記錄
        self.best_distance = float('inf')
        self.worst_distance = 0.0
        self.done_condition = 0.1
        
        # 設定 episode 相關
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_done = False
        self.episode_count = 0
        self.episode_memory = []
        self.max_episode_steps = 1000
        
        # 設定狀態追蹤
        self.prev_states = None
        self.prev_action = None
        self.prev_raw_goal = None
        
        # 設定目的地
        self.destination = {'Lon': 118.22, 'Lat': 24.27}
        
        # 設定初始快照
        self.init_snapshot = {
            'friendly': {},
            'enemy': {}
        }
        
        # 設定重置命令
        self.reset_cmd = ""
        
        # 初始化訓練統計記錄器
        self.stats_logger = Logger()
        
        # 添加步數計數器
        self.step_counter = 0
        
        # 在第一回合取到 unit 時填好：
        self.init_lat, self.init_lon = 23.38, 119.19
        
        # 敵方艦艇初始屬性（第一次 record 用）
        self.enemy_init_lat, self.enemy_init_lon = 24.01, 118.20
        
        # 設定 Args
        class Args:
            def __init__(self):
                self.input_size = 6                # ⇦ 你自己的 state size
                self.manager_hidden_dim = 256
                self.worker_hidden_dim = 256

                # ↓ 這兩行是 FeUdal_agent 必要參數
                self.state_dim_d = 5           #   d   (可設 == input_size 或更小)
                self.embedding_dim_k = 16      #   k   (通常 8~32 都行)

                self.goal_dim = 16
                self.n_actions = 5
                self.manager_dilation = 10
                self.entropy_coef = 0.02
                self.gamma = 0.99
                self.gae_lambda = 0.95
                self.max_grad_norm = 1.0
        
        self.args = Args()
        self.input_size = self.args.input_size        # ← 新增
        
        # 建立網路
        self.manager = Feudal_ManagerAgent(self.input_size, self.args).to(self.device)
        self.worker = Feudal_WorkerAgent(self.input_size, self.args).to(self.device)
        self.manager_critic = FeUdalCritic(self.input_size, self.args).to(self.device)
        self.worker_critic = FeUdalCritic(self.input_size, self.args).to(self.device)
        
        # 建立優化器
        self.optimizer = optim.Adam(
            list(self.manager.parameters()) +
            list(self.worker.parameters()) +
            list(self.manager_critic.parameters()) +
            list(self.worker_critic.parameters()),
            lr=3e-4
        )
        
        # 初始化 LSTM 隱藏狀態
        self.manager_hidden = {self.manager_name: self.manager.init_hidden()}
        self.worker_hidden = {}  # 其他艦動態加入
        
        # 設定最大距離
        self.max_distance = 90.0      # 最大距離限制
        
        # 訓練相關參數
        self.total_steps = 0            # 訓練步計數
        self.gamma = self.args.gamma    # 折扣因子
        self.gae_lambda = self.args.gae_lambda  # GAE lambda

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
    
    def get_distance(self, dx: float, dy: float) -> float:
        """計算標準化距離"""
        return np.sqrt(dx**2 + dy**2)

    def get_extrinsic_reward(self, s, a, ns, done):
        """計算外在獎勵"""
        # 使用狀態向量中的距離值
        distance = s[0]  # dist_norm
        next_distance = ns[0]
        
        # 計算距離變化
        distance_change = distance - next_distance
        
        # 根據距離變化給予獎勵
        if distance_change > 0:
            reward = 1.0  # 距離減少，給予正向獎勵
        elif distance_change < 0:
            reward = -1.0  # 距離增加，給予負向獎勵
        else:
            reward = 0.0  # 距離不變，無獎勵
            
        # 如果達到目標，給予額外獎勵
        if next_distance < self.done_condition:
            reward += 10.0
            
        return reward

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
        # 若沒傳 friendly_names，自動建立順序
        if not self.friendly_order:
            self.friendly_order = [u.Name for u in features.units if u.Side == self.player_side]
        
        # 建立每艦 worker_hidden
        for n in self.friendly_order:
            if n not in self.worker_hidden:
                self.worker_hidden[n] = self.worker.init_hidden()

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
                    'Mission': ENEMY_MISSION        # ← 強制寫死
                }
            # 若 enemy 仍空，下一個 observation 再補
            if not self.init_snapshot['enemy'] and enemy_units:
                for u in enemy_units:
                    self.init_snapshot['enemy'][u.Name] = {
                        'Lat': float(u.Lat),
                        'Lon': float(u.Lon),
                        'DBID': u.DBID,
                        'Mission': ENEMY_MISSION        # ← 強制寫死
                    }
            self.logger.info("已記錄初始艦艇狀態快照")
        
        # 記錄敵方陣營名稱
        if self.enemy_side is None:
            self.enemy_side = enemy_side
        
        print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        ac = self.get_unit_info_from_observation(features, self.manager_name)
        
        # 如果找不到單位，可能是被擊沉或其他原因
        if ac is None:
            # 確保我們已經記錄過船隻類型和 ID（至少有初始值）
            if self.ship_type is None or self.ship_dbid is None:
                self.logger.error(f"單位消失且尚未記錄船隻類型和 ID，無法重生")
                return ""  # 返回空命令
            
            self.logger.warning(f"找不到單位: {self.manager_name}，可能已被擊沉，執行重生")
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
        current_states = self.get_states(features)   # list 長度 A，每元素 shape=[F]
        
        # Manager forward
        idx_mgr = self.friendly_order.index(self.manager_name)
        state_mgr = torch.tensor(current_states[idx_mgr], dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_raw, (h_m, c_m), raw_goal = self.manager(state_mgr, self.manager_hidden[self.manager_name])
        self.manager_hidden[self.manager_name] = (h_m.detach(), c_m.detach())
        goal_shared = goal_raw.detach()
        self.prev_raw_goal = raw_goal
        
        # 如果有前一步資料，收集經驗
        if self.prev_states is not None and self.prev_action is not None:
            ext_reward_scalar = self.get_extrinsic_reward(
                self.prev_states[idx_mgr],
                self.prev_action[idx_mgr],
                current_states[idx_mgr],
                False
            )
            ext_reward = np.full(len(self.friendly_order), ext_reward_scalar, dtype=np.float32)  # [A]、float32
            done = current_states[idx_mgr][0] < self.done_condition  # 使用 Manager 艦的距離
            self.episode_reward += ext_reward_scalar
            
            # 將經驗存儲到 episode_memory
            self.episode_memory.append((
                self.prev_states,
                self.prev_action,
                ext_reward,
                current_states,
                done,
                self.prev_raw_goal
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
        
        # Worker 迴圈
        actions = []
        action_cmd = ""
        for name, st in zip(self.friendly_order, current_states):
            h_w, c_w = self.worker_hidden[name]
            logits, (h_w, c_w) = self.worker(
                torch.tensor(st, dtype=torch.float32).unsqueeze(0).to(self.device),
                (h_w, c_w),
                goal_shared
            )
            self.worker_hidden[name] = (h_w.detach(), c_w.detach())
            act = Categorical(logits=logits).sample().item()
            actions.append(act)
            unit = self.get_unit_info_from_observation(features, name)
            action_cmd += "\n" + self.apply_action(
                act, unit,
                contacts,                # 直接傳 contacts list
                VALID_FUNCTIONS          # run-loop 送進來的動作表
            )

        # 每一步都確保敵方單位在巡邏任務中
        for unit in features.get_side_units(self.enemy_side):
            action_cmd += "\n" + set_unit_to_mission(unit.Name, ENEMY_MISSION)

        # 更新前一步資料
        self.prev_states = current_states
        self.prev_action = actions
        self.prev_raw_goal = raw_goal
        
        # 累計步數
        self.episode_step += 1  # 追蹤本回合步數
        self.total_steps += 1

        return action_cmd
    
    def get_state(self, features: FeaturesFromSteam, unit_name: str) -> np.ndarray:
        """
        回傳單一艦艇的狀態向量：[dist_norm, rel_x, rel_y, sinθ, cosθ, enemy_exists]
        """
        # 1) 拿我方船經緯度
        ac = self.get_unit_info_from_observation(features, unit_name)
        if ac is None:
            return np.zeros(6, dtype=np.float32)

        lon_B, lat_B = float(ac.Lon), float(ac.Lat)

        # 2) 拿金門目的地經緯度
        lon_D = float(self.destination['Lon'])
        lat_D = float(self.destination['Lat'])

        # 3) normalization 範圍
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

        # 6) 計算標準化距離
        dist_norm = self.get_distance(rel_x, rel_y)

        # 7) 方位角 θ
        θ = math.atan2(rel_y, rel_x)
        sinθ = math.sin(θ)
        cosθ = math.cos(θ)

        # 8) 偵測敵艦是否存在
        exists = 0.0
        if self.target_name:
            contact = self.get_contact_info_from_observation(features, self.target_name)\
                      or self.get_unit_info_from_observation(features, self.target_name)
            if contact:
                exists = 1.0

        # 9) 組成 state 並回傳
        state = np.array([dist_norm, rel_x, rel_y, sinθ, cosθ, exists], dtype=np.float32)
        return state

    def get_states(self, features: FeaturesFromSteam) -> list[np.ndarray]:
        """
        回傳所有友軍艦艇的狀態列表
        """
        return [self.get_state(features, name) for name in self.friendly_order]

    def apply_action(self, action: int, unit: Unit,
                     contacts, vf) -> str:
        """
        根據動作編號生成對應的 CMO 命令。
        
        參數:
            action: 動作編號 (0-4)
            unit: 要控制的單位
            contacts: 接觸點列表
            vf: 可用函數列表
            
        返回:
            CMO 命令字串
        """
        # 1) 先處理 0-3 四個移動命令
        if action == 0:   # 北
            return set_unit_heading_and_speed(self.player_side, unit.Name, 0, 30)
        elif action == 1: # 東
            return set_unit_heading_and_speed(self.player_side, unit.Name, 90, 30)
        elif action == 2: # 南
            return set_unit_heading_and_speed(self.player_side, unit.Name, 180, 30)
        elif action == 3: # 西
            return set_unit_heading_and_speed(self.player_side, unit.Name, 270, 30)

        # 2) action == 4 → 攻擊最近的敵艦
        if action == 4:
            target = None
            min_dist = float('inf')
            for c in contacts:
                # 計算與敵艦的距離
                dx = float(c.get('Lon', 0)) - float(unit.Lon)
                dy = float(c.get('Lat', 0)) - float(unit.Lat)
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    target = c

            if target and 'ID' in target:
                fn_list = [f for f in vf if f.name == "auto_attack_contact"]
                if fn_list:
                    fn = fn_list[0]
                    return fn.corresponding_def(unit.ID, target['ID'])
            return ""

        return ""

    def train_on_episode(self):
        """使用當前 episode 的數據進行 on-policy 訓練"""
        # 防止空 memory 的 guard
        if not self.episode_memory:
            return

        # 解包
        states, actions, ext, next_states, dones, raw_goals = zip(*self.episode_memory)
        T = len(states)  # 時間步數
        A = len(self.friendly_order)  # 艦艇數量
        feat_dim = self.args.input_size  # 特徵維度
        idx_mgr = self.friendly_order.index(self.manager_name)  # Manager 艦的索引

        # 轉 tensor (使用更高效的方式)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)       # [T,A,feat_dim]
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)  # [T,A,feat_dim]
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)        # [T,A]
        ext_rewards = torch.tensor(ext, dtype=torch.float32, device=self.device)      # [T,A]
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)      # [T]
        raw_goals_mgr = torch.stack(raw_goals).squeeze(1)      # Manager 用，保留梯度
        raw_goals_wrk = raw_goals_mgr.detach()                 # Worker/Intrinsic 用，斷梯度

        # ------------ Worker ------------
        # 計算所有艦艇的價值
        v_w = self.worker_critic(states.reshape(-1, feat_dim)).view(T, A)  # [T,A]
        with torch.no_grad():
            nv_w = self.worker_critic(next_states.reshape(-1, feat_dim)).view(T, A)  # [T,A]

        # 計算內在獎勵並廣播到所有艦艇
        intr = self.compute_intrinsic_rewards(states[:, idx_mgr], raw_goals_wrk, c=self.args.manager_dilation)
        intr = intr.unsqueeze(1).repeat(1, A)   # [T-c, A]

        # 組合獎勵
        dr = 2 * intr + 0.8 * ext_rewards[self.args.manager_dilation:]  # [T-c, A]

        # GAE for Worker
        deltas_w = dr + self.gamma * nv_w[self.args.manager_dilation:] * (1 - dones[self.args.manager_dilation:].unsqueeze(1)) - v_w[self.args.manager_dilation:]
        advantages_w = torch.zeros_like(dr, device=self.device)
        last_adv_w = 0.0
        for t in reversed(range(len(dr))):
            done_t = dones[t + self.args.manager_dilation]     # scalar tensor
            last_adv_w = deltas_w[t] + self.gamma * self.gae_lambda * \
                         (1 - done_t) * last_adv_w            # 直接乘，會自動 broadcast
            advantages_w[t] = last_adv_w

        # Worker Policy Loss
        flat_states = states[self.args.manager_dilation:].reshape(-1, feat_dim)
        flat_TA = flat_states.size(0)  # T' * A
        zeros = (
            torch.zeros(flat_TA, self.args.worker_hidden_dim, device=self.device),
            torch.zeros(flat_TA, self.args.worker_hidden_dim, device=self.device)
        )
        # 取 Manager 的 raw_goal（T', K）並 broadcast 到每艦
        raw_goal_exp = raw_goals_wrk[self.args.manager_dilation:].repeat_interleave(A, dim=0)  # [T'*A, K]
        logits_w, _ = self.worker(flat_states, zeros, raw_goal_exp)
        logits_w = logits_w.view(-1, A, self.args.n_actions)
        log_probs_w = F.log_softmax(logits_w, dim=-1)
        action_log_probs_w = log_probs_w.gather(-1, actions[self.args.manager_dilation:].unsqueeze(-1)).squeeze(-1)
        policy_loss_w = -(action_log_probs_w * advantages_w.detach()).mean()

        # Worker Value Loss
        value_loss_w = 0.5 * (deltas_w.detach() ** 2).mean()

        # ------------ Manager ------------
        # 只使用 Manager 艦的狀態
        v_m = self.manager_critic(states[:, idx_mgr])  # [T,1]
        with torch.no_grad():
            nv_m = self.manager_critic(next_states[:, idx_mgr])  # [T,1]

        # GAE for Manager
        deltas_m = ext_rewards[:, idx_mgr] + self.gamma * nv_m.squeeze(-1) * (1 - dones) - v_m.squeeze(-1)
        advantages_m = torch.zeros_like(deltas_m, device=self.device)
        last_adv_m = 0.0
        for t in reversed(range(len(deltas_m))):
            done_t = dones[t]                                  # scalar tensor
            last_adv_m = deltas_m[t] + self.gamma * self.gae_lambda * \
                         (1 - done_t) * last_adv_m
            advantages_m[t] = last_adv_m

        # Manager Policy Loss (使用 cosine similarity)
        delta = states[self.args.manager_dilation:, idx_mgr] - states[:-self.args.manager_dilation, idx_mgr]
        delta_emb = self.manager.delta_fc(delta)
        cos_term = F.cosine_similarity(
            F.normalize(delta_emb, dim=-1, p=2),
            F.normalize(raw_goals_mgr[:-self.args.manager_dilation], dim=-1, p=2),
            dim=-1
        ) + 1      # +1 保持正值

        policy_loss_m = -(advantages_m[:-self.args.manager_dilation].detach() * cos_term).mean()

        # Manager Value Loss
        value_loss_m = 0.5 * (deltas_m.detach() ** 2).mean()

        # ------------ 總 Loss ------------
        total_loss = (
            policy_loss_w + 
            value_loss_w + 
            policy_loss_m + 
            value_loss_m
        )

        # 更新參數
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.manager_critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.optimizer.step()

        # 記錄
        self.logger.info(
            f"Episode {self.episode_count} 訓練完成:\n"
            f"  Worker Policy Loss: {policy_loss_w.item():.4f}\n"
            f"  Worker Value Loss: {value_loss_w.item():.4f}\n"
            f"  Manager Policy Loss: {policy_loss_m.item():.4f}\n"
            f"  Manager Value Loss: {value_loss_m.item():.4f}\n"
            f"  Total Loss: {total_loss.item():.4f}"
        )

        # 記錄統計數據
        self.stats_logger.log_stat("total_worker_loss", policy_loss_w.item(), self.total_steps)
        self.stats_logger.log_stat("total_manager_loss", policy_loss_m.item(), self.total_steps)
        self.stats_logger.log_stat("extrinsic_reward", ext_rewards.mean().item(), self.total_steps)
        self.stats_logger.log_stat("intrinsic_reward", intr.mean().item(), self.total_steps)
        self.stats_logger.log_stat("return", self.episode_reward, self.total_steps)
        
        # 記錄最終距離
        final_state = next_states[-1].cpu().numpy()
        idx_mgr = self.friendly_order.index(self.manager_name)
        final_norm_dist = float(final_state[idx_mgr][0])  # Manager 艦 dist_norm
        self.stats_logger.log_stat("final_norm_dist", final_norm_dist, self.total_steps)
        self.logger.info(f"Final normalized distance: {final_norm_dist:.4f}")
    
    def build_reset_cmd(self) -> str:
        """建立重置命令"""
        reset_cmd = ""
        
        # ---------- 友方 ----------
        for name, info in self.init_snapshot['friendly'].items():
            reset_cmd += delete_unit(side=self.player_side, unit_name=name) + "\n"
            reset_cmd += add_unit(
                type='Ship', unitname=name, dbid=info['DBID'],
                side=self.player_side, Lat=info['Lat'], Lon=info['Lon']
            ) + "\n"
            # 若原本就有任務可以保留
            if info['Mission']:
                reset_cmd += set_unit_to_mission(name, info['Mission']) + "\n"

        # ---------- 敵方 ----------
        for name, info in self.init_snapshot['enemy'].items():
            reset_cmd += delete_unit(side=self.enemy_side, unit_name=name) + "\n"
            reset_cmd += add_unit(
                type='Ship', unitname=name, dbid=info['DBID'],
                side=self.enemy_side, Lat=info['Lat'], Lon=info['Lon']
            ) + "\n"
            # 無論原來 mission 為何，一律派回 5VS5
            reset_cmd += set_unit_to_mission(name, ENEMY_MISSION) + "\n"

        self.reset_cmd = reset_cmd.strip()
        return self.reset_cmd

    def reset(self) -> str:
        """重置環境"""
        # 重置 episode 相關變數
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_done = False
        self.episode_count += 1
        self.episode_memory = []
        
        # 重置距離記錄
        self.best_distance = float('inf')
        self.worst_distance = 0.0   # 重新起算最差距離
        
        # 重置狀態追蹤
        self.prev_states = None
        self.prev_action = None
        self.prev_raw_goal = None
        
        # 重置 Manager 隱藏狀態
        self.manager_hidden = {self.manager_name: self.manager.init_hidden()}
        
        # 重置 Worker 隱藏狀態
        for name in self.friendly_order:
            self.worker_hidden[name] = self.worker.init_hidden()
        
        # 返回重置命令
        return self.reset_cmd