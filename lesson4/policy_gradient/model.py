import numpy as np
import torch
from time import sleep
import torch.nn.functional as F


class Model(torch.nn.Module):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid1_size = act_dim * 10
        self.fc1 = torch.nn.Linear(obs_dim, hid1_size)
        self.fc2 = torch.nn.Linear(hid1_size, act_dim)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        obs = self.fc1(obs)
        obs = self.tanh(obs)
        obs = self.fc2(obs)
        prob = self.softmax(obs)
        return prob

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    a = torch.Tensor(np.random.randn(48))
    mo = Model(48, 4)
    print(mo(a))
