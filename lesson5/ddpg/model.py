'''
使用pytorch搭建dqn中的神经网络
'''

import numpy as np
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        self.actor_model = Actor(obs_dim, act_dim)
        self.critic_model = Critic(obs_dim, act_dim)

    def act(self, obs):
        return self.actor_model(obs)

    def criticize(self, obs, action):
        return self.critic_model(obs, action)


class Actor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):

        super(Actor, self).__init__()
        self.hid1_size = 128
        # 3层全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, self.hid1_size),  # obs_dim --> 128
            torch.nn.ReLU(),
            torch.nn.Linear(self.hid1_size, self.hid1_size),  # 128 --> 128
            torch.nn.ReLU(),
            torch.nn.Linear(self.hid1_size, act_dim),  # 128 --> act_dim
        )

    def forward(self, obs):
        obs = self.fc(obs)
        means = torch.tanh(obs)
        return means


class Critic(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        hid_size = 100

        self.l1 = torch.nn.Linear(obs_dim + act_dim, hid_size)
        self.l2 = torch.nn.Linear(hid_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, obs, act):
        obs_act = torch.cat([obs, act], dim=1)
        obs_act = self.relu(self.l1(obs_act))
        Q = self.l2(obs_act)
        return Q
