import numpy as np
import torch
from time import sleep
import torch.nn.functional as F
import torch.nn as nn

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

    def forward(self, obs):
        obs = self.fc1(obs)
        obs = self.tanh(obs)
        obs = self.fc2(obs)
        prob = self.softmax(obs)
        return prob

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class FCN_model(torch.nn.Module):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(FCN_model, self).__init__()
        hid1_size = 256
        hid2_size = 64

        self.fc1 = torch.nn.Linear(obs_dim, hid1_size)
        self.fc2 = torch.nn.Linear(hid1_size, hid2_size)
        self.fc3 = torch.nn.Linear(hid2_size, act_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, obs):
        obs = self.fc1(obs)
        obs = self.relu(obs)
        obs = self.fc2(obs)
        obs = self.relu(obs)
        obs = self.fc3(obs)
        prob = self.softmax(obs)
        return prob

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class CNN_model(torch.nn.Module):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8,
                      kernel_size=3, stride=1, padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(8*int(obs_dim/16), 512)
        self.fc2 = nn.Sequential(nn.Linear(512, act_dim))
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, obs):
        obs = self.conv1(obs)
        obs = self.conv2(obs)
        obs = obs.view(-1, 3200)
        obs = self.fc1(obs)
        obs = self.relu(obs)
        obs = self.fc2(obs)
        prob = self.softmax(obs).squeeze()

        return prob

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    a = torch.Tensor(np.random.randn(48))
    mo = Model(48, 4)
    print(mo(a))
