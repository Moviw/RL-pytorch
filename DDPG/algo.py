from model import *
import numpy as np


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim,  action_dim, action_up_bound, action_down_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        # self.actor = PolicyNet(state_dim,
        #                        action_dim, action_up_bound,action_down_bound).to(device)
        # self.critic = QValueNet(state_dim,  action_dim).to(device)
        # self.target_actor = PolicyNet(
        #     state_dim,  action_dim, action_up_bound,action_down_bound).to(device)
        # self.target_critic = QValueNet(
        #     state_dim,  action_dim).to(device)
        self.model = DDPG_Net(state_dim, action_dim,
                              action_up_bound, action_down_bound).to(device)
        self.target_model = DDPG_Net(
            state_dim, action_dim, action_up_bound, action_down_bound).to(device)
        # 初始化目标价值/策略网络并设置和价值/策略网络相同的参数
        self.target_model.critic_model.load_state_dict(
            self.model.critic_model.state_dict())
        self.target_model.actor_model.load_state_dict(
            self.model.actor_model.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_model.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.model.act(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(
            transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_model.criticize(
            next_states, self.target_model.act(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(
            self.model.criticize(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - \
            torch.mean(self.model.criticize(states, self.model.act(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.model.actor_model,
                         self.target_model.actor_model)  # 软更新策略网络
        self.soft_update(self.model.critic_model,
                         self.target_model.critic_model)  # 软更新价值网络

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, path):
        self.model.load(path)
