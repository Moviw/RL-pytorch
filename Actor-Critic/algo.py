import torch
from torch.nn import MSELoss
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2C():
    def __init__(self, model, gamma=None, actor_lr=None, critic_lr=None):
        """ A2C algorithm

            Args:
                model (Model): actor and critic 的前向网络.
                gamma (float): reward的衰减因子.
                actor_lr (float): actor 的学习率
                critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model

        self.mse_loss = MSELoss().to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_model.parameters(), lr=actor_lr, )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_model.parameters(), lr=critic_lr, )

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        return self.model.act(obs)

    def learn(self, obs, action, reward, next_obs, done):
        """ 用A2C算法更新 actor 和 critic
        """
        current_V = self.model.criticize(obs)
        next_V=self.model.criticize(next_obs)

        # 时序差分目标
        td_target = reward + self.gamma * next_V * (1 - done)
        td_delta = td_target - current_V  # 时序差分误差

        # 计算 Actor loss
        log_probs = torch.log(self.model.act(obs).gather(1, action))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # 计算 Critic loss
        critic_loss = self.mse_loss(current_V, td_target.detach())

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

        return critic_loss.item(), actor_loss.item()

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, path):
        self.model.load(path)
