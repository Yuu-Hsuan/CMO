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

class RunningNorm:
    """
    線性時間、指數加權的 running mean / std 正規化器
    用法:   x_norm = self.rnorm(x)      # 會同步更新內部 μ, σ
    """
    def __init__(self, momentum=0.999, eps=1e-8, clip=3.0):
        self.m = None            # running mean
        self.v = None            # running var
        self.momentum = momentum
        self.eps = eps
        self.clip = clip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.m is None:          # 第一次看到資料時初始化形狀
            self.m = torch.zeros_like(x.mean(0))
            self.v = torch.ones_like(x.var(0, unbiased=False))

        # 先把 μ/σ 搬到同一裝置
        self.m, self.v = self.m.to(x.device), self.v.to(x.device)

        # 1) 取 batch 統計並立即脫鉤
        b_mean = x.mean(0).detach()
        b_var  = x.var(0, unbiased=False).detach()

        # 2) 用 no_grad 更新 running μ / σ
        with torch.no_grad():
            self.m.mul_(self.momentum).add_((1 - self.momentum) * b_mean)
            self.v.mul_(self.momentum).add_((1 - self.momentum) * b_var)
            
        # 3) 再拿最新 μ / σ 去標準化
        x_hat = (x - self.m) / torch.sqrt(self.v + self.eps)
        return torch.clamp(x_hat, -self.clip, self.clip)

# -----------------------  常數區  -----------------------
ENEMY_MISSION = "Kinmen patrol"      # ★你的劇本裡真正的任務名稱
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
        self.max_episode_steps = 200
        self.episode_init = True
        self.need_reset = False   # 觸發 reset 的總開關
        #self.waiting_delete = False   # 已送 delete、等 CMO 真正刪除
        
        # 設定狀態追蹤
        self.prev_states = None
        self.prev_action = None
        self.prev_raw_goal = None
        # 追蹤上一步誰還活著
        self.prev_alive_mask = None
        # <── 新增 ──>  記錄最後一次看到的經緯度
        # key = unit name，value = (lon, lat)
        self.last_pos = {}
        
        # 設定目的地
        self.destination = {'Lon': 118.27954108343, 'Lat': 24.333113806906}
        
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
                self.manager_dilation = 10  # Manager 每隔幾步更新一次
                self.entropy_coef = 0.2
                self.gamma = 0.99
                self.gae_lambda = 0.95
                self.max_grad_norm = 1.0
                self.n_agents = 0              # 觀測第一次後再設
                self.manager_input_size = 0    # 同上，＝input_size * n_agents
                self.manager_loss_coef = 5.0     # ★ 新增
        
        self.args = Args()
        self.input_size = self.args.input_size        # ← 新增
        
        # 建立網路
        self.manager = None
        self.worker = None
        self.manager_critic = None
        self.worker_critic = None
        self.opt_policy = None
        self.opt_value = None
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

        # --- reward normalizer（動態共享整個 session）---
        self.ext_norm_ship = RunningNorm()   # 給 per-ship ext
        self.ext_norm_mgr  = RunningNorm()   # 給 manager 專用
        self.intr_norm = RunningNorm()  # 給內在獎勵

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

    # ---------- 取代 team 版，用「每艘」計算 ----------
    def get_ship_extrinsic_reward(self, prev_s, curr_s, done):
        # ① 前進價值
        reward = 20.0 * (prev_s[0] - curr_s[0])   # Δd ∈ [-1,1]

        # ② shaping：越近越多，<self.done_condition 直接 +3
        d = curr_s[0]
        if d < self.done_condition:
            reward += 3.0          # 抵達
        elif d < 0.2:  reward += 2.3
        elif d < 0.3:  reward += 2.1
        elif d < 0.4:  reward += 1.9
        elif d < 0.5:  reward += 1.7
        elif d < 0.6:  reward += 1.5            
        elif d < 0.7:  reward += 1.3
        elif d < 0.8:  reward += 1.1
        elif d < 0.9:  reward += 0.9

        # ③ 全隊抵達一次性 bonus
        if done:
            reward += 10.0
        return float(reward)

    def compute_intrinsic_rewards(self, delta_states, raw_goals):
        """
        delta_states : [T-c, A, F]
        raw_goals    : [T-c, A, k]
        回傳         : [T-c, A]（可正可負）
        """
        return F.cosine_similarity(
            F.normalize(self.manager.delta_fc_local(delta_states), dim=-1),
            F.normalize(raw_goals, dim=-1),
            dim=-1
        )

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
            
            # Policy 只管 Manager + Worker
            self.opt_policy = optim.Adam(
                list(self.manager.parameters()) + list(self.worker.parameters()),
                lr=3e-4)

            # Value 只管兩個 Critic
            self.opt_value  = optim.Adam(
                list(self.manager_critic.parameters()) + list(self.worker_critic.parameters()),
                lr=1e-3, weight_decay=1e-5)
        # -------------------------------------------------------
        
        # 建立每艦 worker_hidden
        for n in self.friendly_order:
            if n not in self.worker_hidden:
                self.worker_hidden[n] = self.worker.init_hidden()

        # Manager forward
        manager_state = self.build_manager_state(current_states)          # [A*F]
        manager_state = torch.tensor(manager_state, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)      # [1, A*F]
        
        need_new_goal = (
            not hasattr(self.manager, "current_goal") or
            self.manager.current_goal is None or
            (self.episode_step % self.args.manager_dilation == 0)
        )

        if need_new_goal:
            goal_mat, (h_m, c_m), raw_goal = \
                self.manager(manager_state, self.manager_hidden[self.manager_name])

            # ------- ① 立刻斷開計算圖 -------★
            self.prev_raw_goal = raw_goal.detach()

            # 如果還想保留原 tensor 可用 clone().detach()
            # self.prev_raw_goal = raw_goal.clone().detach()

            self.manager.current_goal = goal_mat.detach()
        else:
            goal_mat = self.manager.current_goal
            h_m, c_m  = self.manager_hidden[self.manager_name]
            raw_goal  = self.prev_raw_goal

        # 記回隱藏狀態 (務必 detach)
        self.manager_hidden[self.manager_name] = (h_m.detach(), c_m.detach())
        
        # 如果有前一步資料，收集經驗
        if self.prev_states is not None and self.prev_action is not None:
            # ------------- 建立本步的 reward -----------------
            alive_now = alive_mask
            if self.prev_alive_mask is None:        # 第一個 time-step
                self.prev_alive_mask = alive_now

            per_ship_reward = []
            for i, (ps, cs) in enumerate(zip(self.prev_states, current_states)):
                # 只要這一步船已死亡 → reward = 0
                if not alive_now[i]:
                    per_ship_reward.append(0.0)
                else:
                    per_ship_reward.append(
                        self.get_ship_extrinsic_reward(ps, cs, success))

            # 更新，供下一步辨識「剛死掉」的船
            self.prev_alive_mask = alive_now
            ext_reward = np.asarray(per_ship_reward, dtype=np.float32)   # [A]

            # 回合統計用平均較直觀；若想累加總和改成 .sum()
            self.episode_reward += float(ext_reward.mean())
            
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
            if success or timeout:
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

            unit = self.get_unit_info_from_observation(features, name)
            if unit is None or name in self.arrived_ships:
                actions.append(-1)
                continue

            actions.append(act)
            action_cmd += "\n" + self.apply_action(act, unit, features)

        # 把停船指令加到回傳字串最前
        action_cmd = action_cmd + stop_cmd        # 讓鎖速指令最後出手

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

        # -------- 取得經緯度 --------
        if ac is not None:                                  # 船還活著
            lon_B, lat_B = float(ac.Lon), float(ac.Lat)
            # 把最新位置寫進 self.last_pos
            self.last_pos[unit_name] = (lon_B, lat_B)
        else:                                               # 船已沉
            # 若曾經看過就用最後座標；否則保持原本行為→全 0
            if unit_name in self.last_pos:
                lon_B, lat_B = self.last_pos[unit_name]
            else:
                return np.zeros(6, dtype=np.float32)

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
        # -------- 0) 先更新熵係數 (退火) ------------
        # 每「結束一個 episode」才乘 0.995；最低 0.005
        self.args.entropy_coef = max(0.02, self.args.entropy_coef * 0.99)

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
        # ext_rewards = torch.tensor(ext, dtype=torch.float32, device=self.device)      # [T,A]
        ext_rewards = torch.from_numpy(np.asarray(ext, np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)      # [T]
        raw_goals_tensor = torch.stack(raw_goals).squeeze(1)      # [T, A, k]
        idx_mgr = self.friendly_order.index(self.manager_name)

        # ------- ② Manager / Worker 全部 detach -------★
        raw_goals_mgr = raw_goals_tensor[:, idx_mgr].detach()   # [T,k]
        raw_goals_wrk = raw_goals_tensor.detach()               # [T,A,k]

        # reshape 全艦隊狀態
        states_flat      = states.reshape(T, -1)        # [T, A*F]
        next_states_flat = next_states.reshape(T, -1)

        # ------------ Worker ------------
        # 計算所有艦艇的價值
        v_w = self.worker_critic(states.reshape(-1, feat_dim)).view(T, A)  # [T,A]
        with torch.no_grad():
            nv_w = self.worker_critic(next_states.reshape(-1, feat_dim)).view(T, A)  # [T,A]

        # 計算內在獎勵並廣播到所有艦艇
        c = self.args.manager_dilation
        delta_states = states[c:] - states[:-c]      # [T-c, A, F]
        raw_goals_aligned = raw_goals_wrk[:-c]       # [T-c, A, k]
        intr = self.compute_intrinsic_rewards(delta_states, raw_goals_aligned) # [T-c, A]

        # 先把未對齊的外在 reward 取出 (T-c, A)
        ext_aligned = ext_rewards[self.args.manager_dilation:]           # [T-c, A]

        # 1) 動態標準化
        ext_std = self.ext_norm_ship(ext_aligned)     # [T-c, A]
        intr_std = self.intr_norm(intr)          # [T-c, A]

        # 2) 直接相加；此時兩者都 ≈ 0±1，無需乘係數
        dr = intr_std + ext_std                  # [T-c, A]

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
        valid_advantages = (valid_advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-6)

        # 加入 entropy 正则后
        dist_w = Categorical(logits=logits_w)            # 重新用 logits 建分布
        entropy_w = dist_w.entropy().mean()                   # 平均熵
        policy_loss_w = -(valid_log_probs * valid_advantages).mean() if valid_log_probs.numel() > 0 else torch.tensor(0.0, device=self.device)
        policy_loss_w = policy_loss_w - self.args.entropy_coef * entropy_w

        # Worker Value Loss
        value_loss_w = 0.5 * (deltas_w ** 2).mean()

        # ------------ Manager ------------
        # 只使用 Manager 艦的狀態
        v_m = self.manager_critic(states_flat)  # [T,1]
        with torch.no_grad():
            nv_m = self.manager_critic(next_states_flat)  # [T,1]

        # GAE for Manager
        # ───── ① 先把對齊後的長度計好 ─────
        c_mgr   = self.args.manager_dilation
        Tprime  = T - c_mgr

        # ───── ② 取出對齊後的外在回饋，再標準化 ─────
        ext_mgr_aligned = ext_rewards[c_mgr:, idx_mgr]
        ext_mgr_std     = self.ext_norm_mgr(ext_mgr_aligned)

        # ───── ③ 同步切割 v / nv / dones ─────
        v_m_aligned   = v_m.squeeze(-1)[c_mgr:]
        nv_m_aligned  = nv_m.squeeze(-1)[c_mgr:]
        dones_aligned = dones[c_mgr:]

        # ───── ④ 用對齊後的張量計 deltas_m ─────
        deltas_m = ext_mgr_std + self.gamma * nv_m_aligned * (1 - dones_aligned) - v_m_aligned
        advantages_m = torch.zeros(Tprime, device=self.device)
        last_adv_m   = 0.0
        for t in reversed(range(Tprime)):
            done_t     = dones_aligned[t]
            last_adv_m = deltas_m[t] + self.gamma * self.gae_lambda * (1 - done_t) * last_adv_m
            advantages_m[t] = last_adv_m

        # 標準化 manager advantage
        advantages_m = (advantages_m - advantages_m.mean()) / (advantages_m.std() + 1e-6)

        # Manager Policy Loss (使用 cosine similarity)
        delta = states_flat[c_mgr:] - states_flat[:-c_mgr]
        delta_emb = self.manager.delta_fc_global(delta)
        cos_term = F.cosine_similarity(
            F.normalize(delta_emb, dim=-1, p=2, eps=1e-8),
            F.normalize(raw_goals_mgr[:-c_mgr], dim=-1, p=2, eps=1e-8),
            dim=-1
        ) + 1      # +1 保持正值

        policy_loss_m = -(advantages_m.detach() * cos_term).mean()

        # Manager Value Loss
        value_loss_m = 0.5 * (deltas_m ** 2).mean()

        λm = self.args.manager_loss_coef
        total_loss  = (
            10 * policy_loss_w + value_loss_w +    # Worker 分支
            λm * policy_loss_m + value_loss_m      # Manager 分支
        )

        self.opt_policy.zero_grad()
        self.opt_value.zero_grad()

        total_loss.backward()          # ★ 只做一次 backward，圖立刻釋放

        torch.nn.utils.clip_grad_norm_(self.manager.parameters(),        self.args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(),         self.args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.manager_critic.parameters(), self.args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(),  self.args.max_grad_norm)

        self.opt_policy.step()           # 仍舊 Actor 專用
        self.opt_value.step()            # 仍舊 Critic 專用

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
        self.stats_logger.log_stat("extrinsic_reward", ext_std.mean().item(), self.total_steps)
        self.stats_logger.log_stat("intrinsic_reward", intr_std.mean().item(), self.total_steps)
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
        self.prev_alive_mask = None
        self.last_pos.clear()

        # -------- Manager / Worker 隱藏狀態 --------
        if self.manager is not None:
            self.manager_hidden[self.manager_name] = self.manager.init_hidden()
            self.manager.current_goal = None
        if self.worker is not None:
            for n in self.friendly_order:
                self.worker_hidden[n] = self.worker.init_hidden()

        # -------- 下回 action() 要重新進行回合初始化 --------
        self.episode_init = True

        # -------- 返回重置指令 --------
        return self.reset_cmd