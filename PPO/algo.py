import torch
import numpy as np
from model import *
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma):
        self.model = PPO_Net(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_model.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_model.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.model.act(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(device)
        td_target = rewards + self.gamma * \
            self.model.criticize(next_states) * (1 - dones)
        td_delta = td_target - self.model.criticize(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(device)
        old_log_probs = torch.log(self.model.act(states).gather(1,
                                                                actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.model.act(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.model.criticize(states), td_target.detach()))
            
            # update parameter of actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update paprameter of critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, path):
        self.model.load(path)

