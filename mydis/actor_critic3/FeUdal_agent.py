import torch.nn as nn
import torch.nn.functional as F
import torch

class DilatedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dilation=2):
        super(DilatedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dilation = dilation
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_states_buffer = []
        self.cell_states_buffer = []
    
    def forward(self, x, states):
        h, c = states
        
        # 将当前状态添加到缓冲区 (使用detach避免梯度累積)
        self.hidden_states_buffer.append(h.detach())
        self.cell_states_buffer.append(c.detach())
        
        # 保持缓冲区大小等于膨胀率
        if len(self.hidden_states_buffer) > self.dilation:
            self.hidden_states_buffer.pop(0)
            self.cell_states_buffer.pop(0)
        
        # 如果缓冲区已满，使用膨胀连接的状态
        if len(self.hidden_states_buffer) == self.dilation:
            h_dilated = self.hidden_states_buffer[0]
            c_dilated = self.cell_states_buffer[0]
            h_out, c_out = self.lstm_cell(x, (h_dilated, c_dilated))
        else:
            # 緩衝區未滿時，使用當前狀態
            h_out, c_out = self.lstm_cell(x, (h, c))
        
        return h_out, c_out

class Feudal_ManagerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_ManagerAgent, self).__init__()
        self.args = args

        # Manager network
        self.manager_fc1 = nn.Linear(input_shape, args.manager_hidden_dim)
        # self.manager_rnn = nn.LSTMCell(args.manager_hidden_dim, args.manager_hidden_dim)

        self.manager_rnn = DilatedLSTMCell(args.manager_hidden_dim, args.manager_hidden_dim, dilation=args.manager_dilation)
        # self.manager_fc2 = nn.Linear(args.manager_hidden_dim, args.state_dim_d)

        # 目標生成
        self.goal_network = nn.Linear(args.manager_hidden_dim, args.state_dim_d)
        
        # 狀態值估計 V_t^M
        # self.value_network = nn.Linear(args.manager_hidden_dim, 1)

    def init_hidden(self):
        # Initialize hidden states for both manager and worker
        manager_hidden = self.manager_fc1.weight.new(1, self.args.manager_hidden_dim).zero_()
        manager_cell = self.manager_fc1.weight.new(1, self.args.manager_hidden_dim).zero_()
        return (manager_hidden, manager_cell)

    def forward(self, inputs, hidden):
        features = F.relu(self.manager_fc1(inputs))
        h_in, c_in = hidden
        h_in = h_in.reshape(-1, self.args.manager_hidden_dim)
        c_in = c_in.reshape(-1, self.args.manager_hidden_dim)
        h, c = self.manager_rnn(features, (h_in, c_in))
        
        # 生成目標
        goal = self.goal_network(h)
        
        # 估計狀態值
        # value = self.value_network(h)
        
        return features, goal, (h, c)
    


class Feudal_WorkerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_WorkerAgent, self).__init__()
        self.args = args
        
        # Worker 網絡
        self.worker_fc1 = nn.Linear(input_shape, args.worker_hidden_dim)
        self.worker_rnn = nn.LSTMCell(args.worker_hidden_dim, args.worker_hidden_dim)
        
        # U_t: Action embedding matrix (n_actions x 16)
        self.U_embedding = nn.Linear(args.worker_hidden_dim, args.embedding_dim_k * args.n_actions)
        
        # w_t: 優勢方向/權重 (1x16)
        self.w_network = nn.Linear(args.state_dim_d, args.embedding_dim_k)

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
    
    

