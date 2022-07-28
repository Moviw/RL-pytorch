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
        # 3层全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, act_dim),
        )

    def forward(self, obs):
        obs = self.fc(obs)
        means = torch.tanh(obs)
        return means


class Critic(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(obs_dim + act_dim, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, obs, act):
        obs_act = torch.cat([obs, act], dim=1)
        obs_act = self.fc1(obs_act)
        obs_act = self.relu(obs_act)
        obs_act = self.fc2(obs_act)
        obs_act = self.relu(obs_act)
        Q = self.fc3(obs_act)
        return Q
