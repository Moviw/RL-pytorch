import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim,  action_dim, action_up_bound, action_down_bound):
        super(PolicyNet, self).__init__()
        hidden_dim = 64
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_up_bound = action_up_bound  # action_bound是环境可以接受的动作最大值
        self.action_down_bound = action_down_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * (self.action_up_bound-self.action_down_bound)/2+(self.action_up_bound+self.action_down_bound)/2 # [-1,1] 映射到动作空间 => [down_bound,up_bound]


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim,  action_dim):
        super(QValueNet, self).__init__()
        hidden_dim = 64
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
