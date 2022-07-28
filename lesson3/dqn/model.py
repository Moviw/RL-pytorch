'''
使用pytorch搭建dqn中的神经网络
'''

import numpy as np
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        hid1_size = 128
        # 3层全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hid1_size),  # obs_dim --> 128
            torch.nn.ReLU(),
            torch.nn.Linear(hid1_size, hid1_size),  # 128 --> 128
            torch.nn.ReLU(),
            torch.nn.Linear(hid1_size, act_dim),  # 128 --> act_dim
        )

    def forward(self, obs):
        Q = self.fc(obs)
        return Q


if __name__ == '__main__':
    a = torch.Tensor(np.random.randn(48))
    mo = Net(48, 4)
    print(mo(a))
