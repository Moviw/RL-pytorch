'''
使用pytorch搭建dqn中的神经网络
'''

import numpy as np
import torch
import torch.nn.functional as F


class VANet(torch.nn.Module):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(VANet, self).__init__()
        hidden_dim = 128
        # 3层全连接层
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, act_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - torch.mean(A)  # Q值由V值和A值计算得到
        return Q

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    a = torch.Tensor(np.random.randn(48))
    mo = VANet(48, 4)
    print(mo(a))
