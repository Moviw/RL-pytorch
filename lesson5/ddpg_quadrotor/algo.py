import copy
import torch
from torch.nn import MSELoss
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG():
    def __init__(self, model, gamma=None, actor_lr=None, critic_lr=None):
        """ DDPG algorithm

            Args:
                model (parl.Model): actor and critic 的前向网络.
                                    model 必须实现 get_actor_params() 方法.
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
        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad = False  # 这条居然不写也行 不知道具体原因是什么

        self.mse_loss = MSELoss().to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_model.parameters(), lr=actor_lr, )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_model.parameters(), lr=critic_lr, )

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        return self.model.act(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 用DDPG算法更新 actor 和 critic
        """
        critic_loss = self._critic_learn(
            obs, action, reward, next_obs, terminal)
        actor_loss = self._actor_learn(obs)

        # self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        # 计算 target Q
        target_Q = self.target_model.criticize(
            next_obs, self.target_model.act(next_obs))
        target_Q = reward + self.gamma * target_Q * (1. - terminal)

        # 获取 Q
        current_Q = self.model.criticize(obs, action)

        # 计算 Critic loss
        critic_loss = self.mse_loss(current_Q, target_Q)

        # 优化 Critic 参数
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        # 计算 Actor loss
        actor_loss = -self.model.criticize(obs, self.model.act(obs)).mean()

        # 优化 Actor 参数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    # def sync_target(self, decay=None):
    #     """ self.target_model从self.model复制参数过来，若decay不为None,则是软更新
    #     """
    #     if decay is None:
    #         decay = 1.0 - self.tau
    #     self.model.sync_weights_to(self.target_model, decay=decay)

    def sync_target(self, decay=0.5):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        model_param = self.model.state_dict()
        target_model_param = self.target_model.state_dict()
        for par1, par2 in zip(model_param, target_model_param):
            target_model_param[par2] = model_param[par1] * \
                decay+target_model_param[par2]*(1-decay)
        self.target_model.load_state_dict(target_model_param)
