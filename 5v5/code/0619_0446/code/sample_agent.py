from pycmo.lib.actions import (
    set_unit_position,
    set_unit_heading_and_speed,
    delete_unit,          # ★新增
    add_unit,             # ★新增
    set_unit_to_mission,   # ← 這樣就夠
    manual_attack_contact,    # ← 新增
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
import os

# 导入FeUdal模型
from scripts.FeUdal55_new.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic

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
        self.best_distance = None
        self.worst_distance = 0.0
        self.done_condition = 0.2
        
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
                self.n_actions = 4
                self.manager_dilation = 10
                self.entropy_coef = 0.02
                self.gamma = 0.99
                self.gae_lambda = 0.95
                self.max_grad_norm = 1.0
                self.n_agents = 0              # 觀測第一次後再設
                self.manager_input_size = 0    # 同上，＝input_size * n_agents
        
        self.args = Args()
        self.input_size = self.args.input_size        # ← 新增
        
        # 建立網路
        self.manager = None
        self.worker = None
        self.manager_critic = None
        self.worker_critic = None
        self.optimizer = None
        self.n_agents = 0           # 會在首次 action() 補上
        
        # 初始化 LSTM 隱藏狀態
        self.manager_hidden = {self.manager_name: (
            torch.zeros(1, self.args.manager_hidden_dim),
            torch.zeros(1, self.args.manager_hidden_dim)
        )}
        self.worker_hidden = {}  # 其他艦動態加入
        
        # 設定最大距離
        self.max_distance = 90.0      # 最大距離限制
        
        # 訓練相關參數
        self.total_steps = 0            # 訓練步計數
        self.gamma = self.args.gamma    # 折扣因子
        self.gae_lambda = self.args.gae_lambda  # GAE lambda
        self.arrived_ships = set()    # ▶︎ 追蹤已抵達艦

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

    def compute_intrinsic_rewards(self, states, raw_goals_all, c):
        """
        states:       [T, A*F]  已展平的全艦狀態
        raw_goals_all:[T, A, k] 每艦 raw_goal
        傳回          [T-c, A]  每艦自己的 intrinsic reward
        """
        T, AF = states.shape
        A, k  = raw_goals_all.shape[1], raw_goals_all.shape[2]
        feat_dim = AF // A                   # 單艦特徵維度

        states = states.view(T, A, feat_dim) # [T, A, feat_dim]

        ir_list = []
        for t in range(T - c):
            delta = states[t + c] - states[t]        # [A, feat_dim]
            delta_emb = self.manager.delta_fc_local(delta) # [A, k]

            cos = F.cosine_similarity(
                    F.normalize(delta_emb, dim=-1, p=2, eps=1e-8),
                    F.normalize(raw_goals_all[t], dim=-1, p=2, eps=1e-8),
                    dim=-1)                          # [A]
            ir_list.append(cos.clamp(-1, 1) + 1)     # shift ≥0

        return torch.stack(ir_list, dim=0)           # [T-c, A]

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

        # ------- 1) 生成 alive_mask -------
        alive_mask = [
            self.get_unit_info_from_observation(features, n) is not None
            for n in self.friendly_order
        ]

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
        
        # ---------- 抵達偵測 ----------
        stop_cmd, new_arrivals = "", []
        # 只有在完成第一次移動之後，才開始檢查「抵達」
        if self.episode_step > 0:
            for name, st in zip(self.friendly_order, current_states):
                if st[0] < self.done_condition:
                    if name not in self.arrived_ships:
                        new_arrivals.append(name)
                    self.arrived_ships.add(name)

        # 首次抵達艦：立即設速度 0
        for name in new_arrivals:
            unit = self.get_unit_info_from_observation(features, name)
            if unit:
                # sample_agent.py 也是這麼做的：用 unit.CH 作為 heading
                heading = int(getattr(unit, 'CH', 0))
                stop_cmd += "\n" + set_unit_heading_and_speed(
                    self.player_side,
                    unit.Name,
                    heading,   # 用 CH 屬性
                    0          # 停船
                )

        # success = 真·全队都到达
        success = (len(self.arrived_ships) == len(self.friendly_order))
        # timeout = 步数超上限
        timeout = (self.episode_step >= self.max_episode_steps)
        # done = 两者之一
        done = success or timeout
        
        # --------------- Lazy build nets (首回合) ---------------
        if self.manager is None:
            self.n_agents = len(self.friendly_order)
            self.args.n_agents = self.n_agents
            self.args.manager_input_size = self.args.input_size * self.n_agents
            
            # 重新建立網路
            self.manager = Feudal_ManagerAgent(
                input_size=self.args.manager_input_size,
                args=self.args,
                n_agents=self.n_agents
            ).to(self.device)
            self.manager_hidden[self.manager_name] = self.manager.init_hidden()
            self.worker  = Feudal_WorkerAgent(
                input_shape=self.args.input_size,   # Worker 不變
                args=self.args
            ).to(self.device)
            self.manager_critic = FeUdalCritic(self.args.manager_input_size, self.args).to(self.device)
            self.worker_critic = FeUdalCritic(self.args.input_size, self.args).to(self.device)
            
            # 把舊 optimizer 丟掉，重開一個
            self.optimizer = optim.Adam(
                list(self.manager.parameters()) +
                list(self.worker.parameters()) +
                list(self.manager_critic.parameters()) +
                list(self.worker_critic.parameters()),
                lr=3e-4
            )
        # -------------------------------------------------------
        
        # 建立每艦 worker_hidden
        for n in self.friendly_order:
            if n not in self.worker_hidden:
                self.worker_hidden[n] = self.worker.init_hidden()

        # Manager forward
        manager_state = self.build_manager_state(current_states)          # [A*F]
        manager_state = torch.tensor(manager_state, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)      # [1, A*F]
        goal_mat, (h_m, c_m), raw_goal = \
            self.manager(manager_state, self.manager_hidden[self.manager_name])
        # goal_mat: [1, A, k]
        self.manager_hidden[self.manager_name] = (h_m.detach(), c_m.detach())
        self.prev_raw_goal = raw_goal
        
        # 如果有前一步資料，收集經驗
        if self.prev_states is not None and self.prev_action is not None:
            # ———— ① 平均距离差推进 reward ————
            prev = np.array([s[0] for s,a in zip(self.prev_states, alive_mask) if a], np.float32)
            curr = np.array([s[0] for s,a in zip(current_states, alive_mask) if a], np.float32)
            avg_dist_delta = 200 * (prev - curr).mean() if prev.size > 0 else 0.0

            # ———— ② 抵达比例奖励 ————
            avg_arrived = (curr < self.done_condition).mean() if curr.size > 0 else 0.0

            # ———— ③ 胜利奖励：仅成功时给 ————
            win_bonus = 50.0 if success else 0.0

            # ———— ④ 距离跃进分段加成 ————
            nd = float(curr.mean()) if curr.size > 0 else 1.0
            if   nd < 0.2: jump_bonus = 0.4
            elif nd < 0.3: jump_bonus = 0.3
            elif nd < 0.4: jump_bonus = 0.2
            elif nd < 0.5: jump_bonus = 0.1
            else:          jump_bonus = 0.0

            # ———— ⑤ 刷新最优平均距离时的小奖励 (第一次跳过) ————
            best_bonus = 0.0
            if self.best_distance is not None and nd + 1e-6 < self.best_distance:
                best_bonus = 0.5
            self.best_distance = nd if self.best_distance is None else min(self.best_distance, nd)

            # ———— ⑥ 越界惩罚（示例：如果某个船距离变化异常就惩） ————
            overflow_penalty = 0.0
            # 这里你可以定义"越界"条件，比如经纬度跑出你设定范围
            # if any(out_of_bounds(s) for s in current_states):
            #     overflow_penalty = -50.0

            # ———— ⑦ 时间惩罚：鼓励更快完成 ————
            time_penalty = -0.01

            # ———— 总外在 reward ————
            ext_reward_scalar = (
                avg_dist_delta
              + avg_arrived
              + win_bonus
              + jump_bonus
              + best_bonus
              + overflow_penalty
              + time_penalty
            )

            # 广播到所有 agent
            ext_reward = np.full(len(self.friendly_order),
                                 ext_reward_scalar,
                                 dtype=np.float32)
            self.episode_reward += ext_reward_scalar
            
            # 存经验
            self.episode_memory.append((
                self.prev_states,
                self.prev_action,
                ext_reward,
                current_states,
                done,
                self.prev_raw_goal,
                alive_mask
            ))
            
            # 如果 episode 結束，就更新，然後 clear episode_memory
            if done:
                # 只有 success 才会有 win_bonus
                self.train_on_episode()
                self.episode_memory.clear()

                # 2) 建 reset block（delete+add+mission），只发重置指令
                reset_block = self.build_reset_cmd()

                # 3) 清 internal state：清掉 arrived_ships / prev_states / episode_step…
                self.reset()

                # 4) 只回傳重置指令，让下一次 action() 重新由 FeUdal 网络算移动
                return reset_block
        
        # Worker 迴圈
        goals_per_ship = goal_mat.squeeze(0)        # [A, k]
        actions, action_cmd = [], ""

        for i, (name, st) in enumerate(zip(self.friendly_order, current_states)):
            h_w, c_w = self.worker_hidden[name]
            g_i = goals_per_ship[i].unsqueeze(0)    # [1, k]
            
            # → 仍跑 Worker 前向、取動作
            logits, (h_w, c_w) = self.worker(
                torch.tensor(st, dtype=torch.float32).unsqueeze(0).to(self.device),
                (h_w, c_w), g_i
            )
            self.worker_hidden[name] = (h_w.detach(), c_w.detach())
            act = Categorical(logits=logits).sample().item()
            actions.append(-1 if name in self.arrived_ships else act)

            if name in self.arrived_ships:
                continue   # 不送指令，但 still forward

            unit = self.get_unit_info_from_observation(features, name)
            action_cmd += "\n" + self.apply_action(act, unit, features)

        # 把停船指令加到回傳字串最前
        action_cmd = stop_cmd + action_cmd

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

    def build_manager_state(self, feature_list) -> np.ndarray:
        """
        把 get_states 回傳的 list[A, F] 展平成單一向量 [A*F]。
        """
        return np.concatenate(feature_list, axis=0).astype(np.float32)

    def apply_action(self, action: int, unit: Unit, features: FeaturesFromSteam) -> str:
        # unit 可能為 None
        if unit is None:
            return ""

        # 0=前進、1=左轉、2=右轉、3=攻擊
        if action == 0:
            heading = unit.CH
        elif action == 1:
            heading = unit.CH - 30
        elif action == 2:
            heading = unit.CH + 30
        elif action == 3:
            # 只發射 Hsiung Feng 系列
            contacts = getattr(features, 'contacts', None) or []
            if not contacts:
                heading = unit.CH
            else:
                enemy = random.choice(contacts)
                # 取 mounts
                mounts = getattr(unit, 'Mounts', None) or []
                weapon_id = None
                for mount in mounts:
                    mname = getattr(mount, 'Name', '').lower()
                    if 'hsiung feng' not in mname:
                        continue
                    weapons = getattr(mount, 'Weapons', []) or []
                    if weapons and weapons[0].QuantRemaining > 0:
                        weapon_id = weapons[0].WeaponID
                        break
                if weapon_id is None:
                    heading = unit.CH
                else:
                    # 發射一枚
                    return manual_attack_contact(
                        attacker_id=unit.ID,
                        contact_id=enemy['ID'],
                        weapon_id=weapon_id,
                        qty=1
                    )
        else:
            heading = unit.CH  # 萬一 action 不在 0~3

        # normalize heading
        heading %= 360
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=unit.Name,
            heading=heading,
            speed=30
        )

    def train_on_episode(self):
        """使用當前 episode 的數據進行 on-policy 訓練"""
        # 防止空 memory 的 guard
        if not self.episode_memory:
            return

        # 解包
        # unpack 多帶出的 alive_mask
        states, actions, ext, next_states, dones, raw_goals, masks = zip(*self.episode_memory)
        T = len(states)
        if T <= self.args.manager_dilation:
            self.logger.warning(f"Skip training: episode length {T} ≤ dilation {self.args.manager_dilation}")
            self.episode_memory.clear()
            return

        A = len(self.friendly_order)  # 艦艇數量
        feat_dim = self.args.input_size  # 特徵維度
        idx_mgr = self.friendly_order.index(self.manager_name)  # Manager 艦的索引

        # 先把「一個 time step」轉成 [A,F] 的 ndarray
        states_np = np.stack([np.stack(step, axis=0) for step in states], axis=0)  # [T,A,F]
        next_states_np = np.stack([np.stack(step, axis=0) for step in next_states], axis=0)  # [T,A,F]
        
        # 轉 tensor
        states = torch.from_numpy(states_np).float().to(self.device)       # [T,A,feat_dim]
        next_states = torch.from_numpy(next_states_np).float().to(self.device)  # [T,A,feat_dim]
        
        # 確保形狀正確
        assert states.shape[1] == len(self.friendly_order), f"States shape mismatch: {states.shape[1]} != {len(self.friendly_order)}"
        
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)        # [T,A]
        ext_rewards = torch.tensor(ext, dtype=torch.float32, device=self.device)      # [T,A]
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)      # [T]
        raw_goals_tensor = torch.stack(raw_goals).squeeze(1)      # [T, A, k]
        idx_mgr = self.friendly_order.index(self.manager_name)

        raw_goals_mgr = raw_goals_tensor[:, idx_mgr]              # [T, k]
        raw_goals_wrk = raw_goals_tensor.detach()                 # [T, A, k]

        # reshape 全艦隊狀態
        states_flat      = states.reshape(T, -1)        # [T, A*F]
        next_states_flat = next_states.reshape(T, -1)

        # ------------ Worker ------------
        # 計算所有艦艇的價值
        v_w = self.worker_critic(states.reshape(-1, feat_dim)).view(T, A)  # [T,A]
        with torch.no_grad():
            nv_w = self.worker_critic(next_states.reshape(-1, feat_dim)).view(T, A)  # [T,A]

        # 計算內在獎勵並廣播到所有艦艇
        intr = self.compute_intrinsic_rewards(
           states_flat, raw_goals_wrk, c=self.args.manager_dilation)  # [T-c, A]

        # 組合獎勵
        dr = 1.0 * intr + 1.0 * ext_rewards[self.args.manager_dilation:]  # [T-c, A]

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
        raw_goal_exp = raw_goals_wrk[self.args.manager_dilation:].reshape(-1, self.args.goal_dim)
        logits_w, _ = self.worker(flat_states, zeros, raw_goal_exp)
        Tprime = states[self.args.manager_dilation:].size(0)
        logits_w = logits_w.view(Tprime, A, self.args.n_actions)
        log_probs_w = F.log_softmax(logits_w, dim=-1)

        actions_t_prime = actions[self.args.manager_dilation:]
        mask = (actions_t_prime != -1)
        safe_actions = actions_t_prime.clone()
        safe_actions[~mask] = 0  # mask out -1 for gather

        action_log_probs_w = log_probs_w.gather(-1, safe_actions.unsqueeze(-1)).squeeze(-1)
        
        # apply mask to loss calculation
        valid_log_probs = action_log_probs_w[mask]
        valid_advantages = advantages_w.detach()[mask]
        
        # 標準化 worker advantage
        valid_advantages = (valid_advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)

        # 加入 entropy 正则后
        dist_w = Categorical(logits=logits_w)            # 重新用 logits 建分布
        entropy_w = dist_w.entropy().mean()                   # 平均熵
        policy_loss_w = -(valid_log_probs * valid_advantages).mean() if valid_log_probs.numel() > 0 else torch.tensor(0.0, device=self.device)
        policy_loss_w = policy_loss_w - self.args.entropy_coef * entropy_w

        # Worker Value Loss
        value_loss_w = 0.5 * (deltas_w.detach() ** 2).mean()

        # ------------ Manager ------------
        # 只使用 Manager 艦的狀態
        v_m = self.manager_critic(states_flat)  # [T,1]
        with torch.no_grad():
            nv_m = self.manager_critic(next_states_flat)  # [T,1]

        # GAE for Manager
        deltas_m = ext_rewards[:, idx_mgr] + self.gamma * nv_m.squeeze(-1) * (1 - dones) - v_m.squeeze(-1)
        advantages_m = torch.zeros_like(deltas_m, device=self.device)
        last_adv_m = 0.0
        for t in reversed(range(len(deltas_m))):
            done_t = dones[t]                                  # scalar tensor
            last_adv_m = deltas_m[t] + self.gamma * self.gae_lambda * \
                         (1 - done_t) * last_adv_m
            advantages_m[t] = last_adv_m

        # 標準化 manager advantage
        advantages_m = (advantages_m - advantages_m.mean()) / (advantages_m.std() + 1e-8)

        # Manager Policy Loss (使用 cosine similarity)
        delta = states_flat[self.args.manager_dilation:] - states_flat[:-self.args.manager_dilation]
        delta_emb = self.manager.delta_fc_global(delta)
        cos_term = F.cosine_similarity(
            F.normalize(delta_emb, dim=-1, p=2, eps=1e-8),
            F.normalize(raw_goals_mgr[:-self.args.manager_dilation], dim=-1, p=2, eps=1e-8),
            dim=-1
        ) + 1      # +1 保持正值

        policy_loss_m = -(advantages_m[:-self.args.manager_dilation].detach() * cos_term).mean()

        # Manager Value Loss
        value_loss_m = 0.5 * (deltas_m.detach() ** 2).mean()

        # ------------ 總 Loss ------------
        λm = getattr(self.args, 'manager_loss_coef', 5.0)     # 默认 5.0，可在 config 里调
        total_loss = (
            10 * policy_loss_w + 
            value_loss_w + 
            λm * policy_loss_m + 
            value_loss_m
        )

        # 檢查是否有 NaN
        if torch.isnan(total_loss).any():
            self.logger.error("NaN detected in loss computation!")
            self.episode_memory.clear()
            return

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
        # 最後一筆 next_states[-1]，對應 masks[-1]
        final_state   = next_states[-1].cpu().numpy()   # [A,F]
        dists_each    = final_state[:, 0]
        final_mask    = np.array(masks[-1], dtype=bool)   # alive mask of last step
        # 只計算還在場上的艦艇
        alive_dists   = dists_each[final_mask]
        if alive_dists.size > 0:
            avg_norm_dist = float(alive_dists.mean())
        else:
            avg_norm_dist = 0.0
        mgr_idx       = self.friendly_order.index(self.manager_name)
        if final_mask[mgr_idx]:
            mgr_norm_dist = float(dists_each[mgr_idx])
        else:
            mgr_norm_dist = 0.0

        self.stats_logger.log_stat("final_norm_dist",      avg_norm_dist, self.total_steps)
        self.stats_logger.log_stat("final_norm_dist_mgr",  mgr_norm_dist, self.total_steps)

        self.logger.info(
            f"Final normalized distance  Avg(alive): {avg_norm_dist:.4f} | "
            f"Manager(alive={final_mask[mgr_idx]}): {mgr_norm_dist:.4f}"
        )
    
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
        """重置環境（安全版本）"""
        # -------- episode 累計變數 --------
        self.episode_step   = 0
        self.episode_reward = 0
        self.episode_done   = False
        self.episode_count += 1
        self.episode_memory = []
        self.arrived_ships.clear()    # ▶︎ 歸零抵達名單

        # -------- 距離統計 --------
        self.best_distance  = None
        self.worst_distance = 0.0

        # -------- 追蹤上一步 --------
        self.prev_states   = None
        self.prev_action   = None
        self.prev_raw_goal = None

        # -------- Manager / Worker 隱藏狀態 --------
        if self.manager is not None:        # ← Manager 已經建好
            self.manager_hidden[self.manager_name] = self.manager.init_hidden()
            self.manager.goal_buffer.clear()
        else:                               # ← 仍處於 Lazy-build 前
            self.manager_hidden = {}        # 等第一個 action() 再補

        if self.worker is not None:
            for name in self.friendly_order:
                self.worker_hidden[name] = self.worker.init_hidden()

        # -------- 返回重置指令 --------
        return self.reset_cmd