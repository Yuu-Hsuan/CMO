import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import deque

class Feudal_ManagerAgent(nn.Module):
    def __init__(self, input_size, args, n_agents):
        super(Feudal_ManagerAgent, self).__init__()
        self.input_size = input_size
        self.hidden_size = args.manager_hidden_dim
        self.embedding_size = args.embedding_dim_k
        self.c = args.manager_dilation
        self.n_agents = n_agents
        
        # 使用標準 LSTM cell
        self.lstm = nn.LSTMCell(input_size, self.hidden_size)
        
        # 目標生成層：一次產出 n_agents × k  維
        self.goal_fc = nn.Linear(self.hidden_size,
                                 self.n_agents * self.embedding_size)
        
        # 目標緩衝區：只保留最近 c 筆 raw_goal
        self.goal_buffer = deque(maxlen=self.c)
        
        # ① 每艦 6 維 → k     (Worker／intrinsic 用)
        self.delta_fc_local  = nn.Linear(args.input_size, self.embedding_size)

        # ② 全隊 A*F → k      (Manager policy 用)
        self.delta_fc_global = nn.Linear(self.input_size, self.embedding_size)   # self.input_size = args.input_size * n_agents

        # 為了少改其他檔，可保留舊別名
        self.delta_fc = self.delta_fc_local
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def init_hidden(self):
        return (torch.zeros(1, self.hidden_size, device=self.goal_fc.weight.device), 
                torch.zeros(1, self.hidden_size, device=self.goal_fc.weight.device))
    
    def forward(self, x, hidden):
        h, c = self.lstm(x, hidden)                  # h: [1, H]
        
        # raw_goal: [1, n_agents, k]
        raw_goal = self.goal_fc(h).view(
            1, self.n_agents, self.embedding_size)
        
        # 時序疊加 (≤c) 後的最終 goal
        self.goal_buffer.append(raw_goal)            # 留每步 raw
        goals = torch.stack(list(self.goal_buffer))  # [≤c, 1, n_agents, k]
        goal_sum = goals.sum(dim=0)                  # [1, n_agents, k]
        
        # L2-normalize 每條船的向量
        goal_norm = F.normalize(goal_sum, dim=-1, p=2)
        return goal_norm, (h, c), raw_goal           # raw_goal 仍保留給訓練

class Feudal_WorkerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_WorkerAgent, self).__init__()
        self.args = args
        
        # Worker 網絡
        self.worker_fc1 = nn.Linear(input_shape, args.worker_hidden_dim)
        self.worker_rnn = nn.LSTMCell(args.worker_hidden_dim, args.worker_hidden_dim)
        
        # U_t: Action embedding matrix (n_actions x 16)
        self.U_embedding = nn.Linear(args.worker_hidden_dim, args.embedding_dim_k * args.n_actions)
        
        # w_t: 優勢方向/權重 (k x k)
        self.w_network = nn.Linear(args.embedding_dim_k, args.embedding_dim_k)

    def init_hidden(self):
        # Initialize hidden states for both manager and worker
        worker_hidden = self.worker_fc1.weight.new(1, self.args.worker_hidden_dim).zero_()
        worker_cell = self.worker_fc1.weight.new(1, self.args.worker_hidden_dim).zero_()
        return (worker_hidden, worker_cell)
    
    def forward(self, inputs, worker_hidden, goal):
        # Worker RNN
        # = F.relu(self.worker_fc1(inputs))
        # = F.softplus(self.worker_fc1(inputs))
        x = F.relu(self.worker_fc1(inputs))
        h_in, c_in = worker_hidden
        h_in = h_in.reshape(-1, self.args.worker_hidden_dim)
        c_in = c_in.reshape(-1, self.args.worker_hidden_dim)
        h, c = self.worker_rnn(x, (h_in, c_in))
        
        # 生成 U_t (action embedding matrix)
        U_t = self.U_embedding(h)
        U_reshaped = U_t.view(-1, self.args.n_actions, self.args.embedding_dim_k)

        # 生成 w_t (優勢方向)
        w_t = self.w_network(goal)
        w_t_reshaped = w_t.view(-1, self.args.embedding_dim_k, 1)
        
        # 計算 s_t (weighted state) 直接當作 policy logits
        logits = torch.bmm(U_reshaped, w_t_reshaped).squeeze(-1)
        
        return logits, (h, c)
        

class FeUdalCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(FeUdalCritic, self).__init__()
        self.args = args

        # 狀態值估計 V(s) —— 兩層 ReLU + 輸出層
        self.value_network = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, inputs):
        return self.value_network(inputs)
    
    

