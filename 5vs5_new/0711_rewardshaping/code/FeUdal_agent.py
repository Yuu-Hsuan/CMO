import torch.nn as nn
import torch.nn.functional as F
import torch

class Feudal_ManagerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_ManagerAgent, self).__init__()
        self.args = args

        # Manager network
        self.manager_fc1 = nn.Linear(input_shape, args.manager_hidden_dim)
        self.manager_rnn = nn.LSTM(args.manager_hidden_dim, args.manager_hidden_dim, batch_first=False)

        # self.manager_rnn = DilatedLSTMCell(args.manager_hidden_dim, args.manager_hidden_dim, dilation=2)
        # self.manager_fc2 = nn.Linear(args.manager_hidden_dim, args.state_dim_d)

        # Actor：輸出連續動作（goal）的均值 μ
        self.mu_head = nn.Linear(args.manager_hidden_dim, args.n_agents * args.state_dim_d)
        # 全域可學對數標準差 log σ
        self.logstd = nn.Parameter(torch.zeros(args.state_dim_d))
        # --- Critic heads (dual) ---
        self.value_head1 = nn.Linear(args.manager_hidden_dim, 1)
        self.value_head2 = nn.Linear(args.manager_hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1):
        # 初始化 Manager 隱藏與 cell 狀態: [num_layers, batch_size, hidden_dim]
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.args.manager_hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.args.manager_hidden_dim, device=device)
        return (h, c)

    def forward(self, inputs, hidden, force_same_goal=False):
        # inputs: [T, B, feat]
        T, B, feat = inputs.shape
        x = inputs
        # feature transform
        features_flat = F.relu(self.manager_fc1(x))  # [T, B, manager_hidden_dim]
        # RNN forward
        out, (h, c) = self.manager_rnn(features_flat, hidden)  # out: [T, B, manager_hidden_dim]
        
        # --- 連續 policy head ---
        mu = self.mu_head(out)    # [T,B,A*d]
        mu = mu.view(T, B, self.args.n_agents, self.args.state_dim_d)
        std = torch.exp(self.logstd)                  # broadcast 到 [d]

        dist = torch.distributions.Normal(mu, std)
        
        if force_same_goal:
            # 如果 force_same_goal=True，使用均值而不是採樣
            g_flat = mu
            logp_flat = dist.log_prob(g_flat).sum(-1, keepdim=True)  # [T,B,A,1]
        else:
            # 正常採樣
            g_flat = dist.rsample()                   # re-parameter 化，可反向傳遞
            logp_flat = dist.log_prob(g_flat).sum(-1, keepdim=True)  # [T,B,A,1]

        # --- dual critics ---
        v1 = self.value_head1(out)         # [T,B*A,1]
        v2 = self.value_head2(out)         # [T,B*A,1]
        v  = 0.5 * (v1 + v2)               # mean value  (向下相容)

        # --- reshape 回 (T,B,A,⋯) ---
        goal  = g_flat.view(T, B, self.args.n_agents, self.args.state_dim_d)
        logp  = logp_flat.view(T, B, self.args.n_agents)               # 之後算 policy loss
        val   = v.squeeze(-1).view(T, B)      # ← 舊格式
        val12 = (                                    # ← 新增
            v1.squeeze(-1).view(T, B),
            v2.squeeze(-1).view(T, B)
        )
        return goal, logp, val, val12, (h, c)
    

class Feudal_WorkerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_WorkerAgent, self).__init__()
        self.args = args
        
        # Worker 網絡
        self.worker_fc1 = nn.Linear(input_shape + args.state_dim_d, args.worker_hidden_dim)
        self.worker_rnn = nn.LSTM(args.worker_hidden_dim, args.worker_hidden_dim, batch_first=False)
        
        # Actor-Critic heads
        self.actor_head  = nn.Linear(args.worker_hidden_dim, args.n_actions)
        self.critic_head1 = nn.Linear(args.worker_hidden_dim, 1)
        self.critic_head2 = nn.Linear(args.worker_hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1):
        # 初始化 Worker 隱藏與 cell 狀態: [num_layers, batch_size * n_agents, hidden_dim]
        B, A = batch_size, self.args.n_agents
        device = next(self.parameters()).device
        h = torch.zeros(1, B*A, self.args.worker_hidden_dim, device=device)
        c = torch.zeros(1, B*A, self.args.worker_hidden_dim, device=device)
        return (h, c)
    
    def forward(self, inputs, worker_hidden, goal):
        # 拼 goal 再展平
        T, B, A, feat = inputs.shape
        x_in = torch.cat([inputs, goal], dim=-1)      # [T,B,A, feat+d]
        x = x_in.view(T, B*A, feat + self.args.state_dim_d)
        
        # feature transform
        x = F.relu(self.worker_fc1(x))  # [T, B*A, worker_hidden_dim]
        # RNN forward
        out, (h, c) = self.worker_rnn(x, worker_hidden)          # [T,B*A,hid]

        # ----- Actor-Critic -----
        logits = self.actor_head(out).view(T, B, A, self.args.n_actions)

        v1 = self.critic_head1(out).squeeze(-1).view(T, B, A)
        v2 = self.critic_head2(out).squeeze(-1).view(T, B, A)
        v  = 0.5 * (v1 + v2)
        return logits, v, (v1, v2), (h, c)