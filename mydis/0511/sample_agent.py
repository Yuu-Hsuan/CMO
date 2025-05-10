from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed
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

# 导入FeUdal模型
from scripts.FeUdal.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic

class MyAgent(BaseAgent):
    def __init__(self, player_side: str, ac_name: str, target_name: str = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param ac_name: 控制的單位名稱（例如 B 船）
        :param target_name: 目標單位名稱（例如 A 船），可選
        :param log_level: 日誌級別，預設為INFO，可設置為logging.DEBUG啟用詳細日誌
        """
        super().__init__(player_side)
        self.ac_name = ac_name
        self.target_name = target_name  # 用於追蹤特定目標（例如 A 船）
        
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
                self.state_dim_d = 6
                self.embedding_dim_k = 16
                self.n_actions = 24  # 8個方向 × 3種速度
                self.manager_dilation = 10  # 改為 c=10

        self.args = Args()
        self.input_size = 6  # [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        
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
        self.batch_size = 256
        self.update_freq = 100
        self.total_steps = 0
        self.done_condition = 0.1     # 歸一化距離閾值（約 0.05 ~ 0.1）
        self.train_interval = 10
        self.entropy_coef = 0.01      # 熵係數

        # 初始化隐藏状态
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()

        self.single_past_goals = []
        self.batch_past_goals = []

        self.best_distance = 2.0      # 初始設為最大可能距離（√2）√2≈1.414→2.0
        self.total_reward = 0

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None
        self.prev_action = None
        self.prev_goal = None
        self.prev_critic_value = None

        # 初始化訓練統計記錄器（如果有的話）
        self.stats_logger = Logger()
        
        # 添加遊戲結束標記
        self.episode_step = 0
        self.max_episode_steps = 200
        self.episode_done = False
        self.episode_reward = 0
        
        # 添加步數計數器
        self.step_counter = 0

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
    def get_distance(self, state):
        """
        用 Haversine 公式計算兩點之間的地理距離 (km)：
        state 依舊是 normalize 後的 [B_lon,B_lat,B_heading,B_speed,A_lon,A_lat]
        這裡先把 lon/lat 還原回真實值，再算距離
        """
        # 如果是 tensor，先轉 numpy
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # 還原回真實經緯度（對應 normalize_state 的上下界）
        lon_B = state[0] * (122.5 - 121.0) + 121.0
        lat_B = state[1] * (24.5   - 23.5) + 23.5
        lon_A = state[4] * (122.5 - 121.0) + 121.0
        lat_A = state[5] * (24.5   - 23.5) + 23.5

        # Haversine 公式
        R = 6371.0  # 地球半徑 (km)
        φ1, φ2 = math.radians(lat_B), math.radians(lat_A)
        Δφ = math.radians(lat_A - lat_B)
        Δλ = math.radians(lon_A - lon_B)

        a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def get_norm_distance(self, state: np.ndarray) -> float:
        """
        直接在歸一化後的 [B_lon,B_lat, …, A_lon,A_lat] 空間裡
        計算兩點間的歐氏距離（範圍大致在 0…√2）。
        """
        # state 是 normalize_state 的輸出：numpy array 長度 6
        dx = state[4] - state[0]
        dy = state[5] - state[1]
        return np.sqrt(dx*dx + dy*dy)

    def debug_print(self, message):
        """只在調試模式下打印信息"""
        if self.debug_mode:
            print(message)
            
    def get_extrinsic_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """
        計算外在獎勵（基於距離變化和時間懲罰）
        """
        distance = self.get_norm_distance(state)
        next_distance = self.get_norm_distance(next_state)
        
        # 基本獎勵：距離變化
        reward = 30 * (distance - next_distance)
        
        # 如果超過了之前的最佳歸一化距離，就給個小獎勵
        if next_distance + 0.01 < self.best_distance:
            self.best_distance = next_distance
            reward += 0.5

        # 到達 done_condition（歸一化距離閾值）時的大額獎勵
        if next_distance < self.done_condition:
            reward += 50

        # 按接近度再加點小獎勵
        if next_distance < 0.2:
            reward += 0.4
        elif next_distance < 0.3:
            reward += 0.3
        elif next_distance < 0.4:
            reward += 0.2
        elif next_distance < 0.5:
            reward += 0.1

        # 越界懲罰（如果 dx/dy 超出 [-1,1]）
        if abs(state[4] - state[0]) > 1 or abs(state[5] - state[1]) > 1:
            reward -= 50
        
        # 時間懲罰
        reward -= 0.01
        
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
        return torch.stack(intrinsic_rewards).squeeze(-1)  # 確保輸出是 [T]

    def action(self, features: FeaturesFromSteam, VALID_FUNCTIONS: AvailableFunctions) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            self.logger.warning(f"找不到單位: {self.ac_name}")
            return action

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
                return self.reset()
        
        # 選擇動作
        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Manager生成目标
            goal, (h_m, c_m), raw_goal = self.manager(state_tensor, self.manager_hidden)
            self.manager_hidden = (h_m, c_m)
            
            # Worker根据目标选择动作
            logits, self.worker_hidden = self.worker(
                state_tensor, 
                self.worker_hidden,
                goal
            )
            
            # 使用 Categorical 分布采样动作
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        
        # 執行動作
        action_cmd = self.apply_action(action, ac)
        
        # 更新前一步資料
        self.prev_state = current_state
        self.prev_action = action
        self.prev_raw_goal = raw_goal
        self.total_steps += 1
        self.episode_step += 1

        return action_cmd
    
    def normalize_state(self, state):
        min_lon, max_lon = 121.0, 122.5  # 根據你的地圖範圍調整
        min_lat, max_lat = 23.5, 24.5
        min_heading, max_heading = 0.0, 360.0
        min_speed, max_speed = 0.0, 30.0
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
        獲取當前狀態向量，包含自身單位和目標的資訊。
        :return: numpy 陣列，例如 [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        """
        # 獲取自身單位（B 船）的資訊
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            # 如果找不到單位，返回預設狀態
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        # 獲取目標單位（A 船）的資訊
        if self.target_name:
            target = self.get_contact_info_from_observation(features, self.target_name) or \
                    self.get_unit_info_from_observation(features, self.target_name)
            if target:
                # 根據 target 的型別提取經緯度
                if isinstance(target, dict):  # 來自 features.contacts
                    target_lon = float(target.get('Lon', 0.0))
                    target_lat = float(target.get('Lat', 0.0))
                else:  # 來自 features.units (Unit 物件)
                    target_lon = float(target.Lon)
                    target_lat = float(target.Lat)
                # 印出目標座標以便檢查
                print(f"目標座標: Lon={target_lon:.6f}, Lat={target_lat:.6f}")
            else:
                target_lon, target_lat = 0.0, 0.0  # 目標未找到時的預設值
                print("警告: 找不到目標單位!")
        else:   
            target_lon, target_lat = 0.0, 0.0  # 未指定目標時的預設值
            print("警告: 未指定目標單位!")
        
        raw_state = np.array([float(ac.Lon), float(ac.Lat), float(ac.CH), float(ac.CS), target_lon, target_lat])
        normalized_state = self.normalize_state(raw_state)
        
        # 將 NumPy 數組轉換為 PyTorch 張量並返回
        # 返回狀態向量：[自身經度, 自身緯度, 自身航向, 自身航速, 目標經度, 目標緯度]
        return normalized_state
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

    def apply_action(self, action: int, ac: Unit) -> str:
        # 擴展動作空間至8個方向
        step_size = 0.005
        lat, lon = float(ac.Lat), float(ac.Lon)
        headings = [0, 45, 90, 135, 180, 225, 270, 315]  # 8個方向
        heading = headings[action % len(headings)]
        
        # 動態調整速度
        speeds = [15, 25, 35]  # 三種速度選項
        speed = speeds[(action // len(headings)) % len(speeds)]
        
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=self.ac_name,
            heading=heading,
            speed=speed
        )

    def train_on_episode(self):
        """使用當前 episode 的數據進行 on-policy 訓練"""
        # 從 episode_memory 拆欄
        states, actions, ext_rewards, next_states, dones, raw_goals = zip(*self.episode_memory)
        T = len(ext_rewards)
        # 這裡用 dilation 代表 Manager 的目標展開長度
        dilation = self.args.manager_dilation

        # 轉 tensor
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        ext_rewards = torch.tensor(ext_rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        raw_goals = torch.stack(raw_goals).squeeze(1)  # [T, k]

        # --- Worker 部分 ---
        values_W = self.worker_critic(states).squeeze(-1)
        with torch.no_grad():
            next_values_W = self.worker_critic(next_states).squeeze(-1)

        # 計算內在獎勵並對齊切片
        intrinsic_rewards = self.compute_intrinsic_rewards(states, raw_goals, dilation)
        # 平移內在獎勵範圍：從 [-1,1] → [0,2]
        # intrinsic_rewards = intrinsic_rewards + 1.0

        dr = intrinsic_rewards
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
            goal_t = raw_goals[t].unsqueeze(0)
            logits_t, (h_w, c_w) = self.worker(state_t, (h_w, c_w), goal_t)
            dist_t = Categorical(logits=logits_t)
            all_log_probs.append(dist_t.log_prob(actions[t].unsqueeze(0)))
            all_entropy.append(dist_t.entropy())

        log_probs = torch.cat(all_log_probs).squeeze(-1)
        entropy = torch.stack(all_entropy).squeeze(-1)

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
        delta_embs = self.manager.delta_fc(next_states - states)
        cos_term_full = F.cosine_similarity(
            F.normalize(delta_embs, dim=-1, p=2),
            F.normalize(raw_goals, dim=-1, p=2),
            dim=-1
        ) + 1
        cos_term = cos_term_full[dilation:]

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
        self.stats_logger.log_stat("worker_loss", total_worker_loss.item(), self.total_steps)
        self.stats_logger.log_stat("manager_loss", total_manager_loss.item(), self.total_steps)
        self.stats_logger.log_stat("extrinsic_reward", ext_rewards.mean().item(), self.total_steps)
        self.stats_logger.log_stat("intrinsic_reward", intrinsic_rewards.mean().item(), self.total_steps)
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
    
    def reset(self):
        """重置遊戲狀態，準備開始新的episode"""
        print(">>> RESETTING position to (24.04, 122.18)")
        self.best_distance = 2.0
        self.prev_state = None
        self.prev_action = None
        self.prev_goal = None
        self.prev_critic_value = None
        self.episode_step = 0
        self.episode_done = False
        self.episode_reward = 0
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()
        # 清空當前episode的記憶
        self.episode_memory = []
        self.logger.info("重置遊戲狀態，準備開始新的episode")
        
        # 清空 Manager 的目標緩衝區
        self.manager.goal_buffer.clear()
        
        # 重置單位位置
        return set_unit_position(
            side=self.player_side,
            unit_name=self.ac_name,
            latitude=24.04,
            longitude=122.18
        )