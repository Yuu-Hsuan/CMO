import torch.nn as nn
import torch.nn.functional as F
import torch

class Feudal_ManagerAgent(nn.Module):
    def __init__(self, input_size, args):
        super(Feudal_ManagerAgent, self).__init__()
        self.input_size = input_size
        self.hidden_size = args.manager_hidden_dim
        self.embedding_size = args.embedding_dim_k
        self.c = args.manager_dilation
        
        # 使用標準 LSTM cell
        self.lstm = nn.LSTMCell(input_size, self.hidden_size)
        
        # 目標生成層
        self.goal_fc = nn.Linear(self.hidden_size, self.embedding_size)
        
        # 目標緩衝區
        self.goal_buffer = []  # 改用 list 替代 deque
        
        # delta 投影層
        self.delta_fc = nn.Linear(self.input_size, self.embedding_size)
        
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
        # 運行 LSTM
        h, c = self.lstm(x, hidden)
        
        # 生成原始目標
        raw_goal = self.goal_fc(h)
        
        # 將原始目標加入緩衝區
        self.goal_buffer.append(raw_goal)
        
        # 如果緩衝區未滿，使用當前目標
        if len(self.goal_buffer) < self.c:
            goal = raw_goal
        else:
            # 將緩衝區中的目標堆疊並求和
            goals = torch.stack(self.goal_buffer)
            goal = goals.sum(dim=0)
        
        # 正規化目標
        goal = F.normalize(goal, dim=-1, p=2)
        
        return goal, (h, c), raw_goal

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

        # 狀態值估計 V_t^M
        # print("&&&&&&&&&&&&&&&&&&&&&&", args.obs_shape)
        self.value_network = nn.Linear(input_shape, 1)

    def forward(self, inputs):
        # 估計狀態值
        value = self.value_network(inputs)
        return value
    
    

