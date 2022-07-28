import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)

        self.act_dim = act_dim
        self.expl_noise = expl_noise
        self.algo = algorithm

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = torch.Tensor(obs).to(device)
        predict_q = self.algo.predict(obs)
        act = predict_q.cpu().detach().numpy()[0]
        act = act.clip(-1, 1)
        return act

    def learn(self, obs, action, reward, next_obs, done):
        """ 根据训练数据更新一次模型参数
        """

        self.algo.sync_target()

        obs = torch.tensor(obs, requires_grad=True).to(device)
        action = torch.tensor(action).reshape(-1, self.act_dim).to(device)
        reward = torch.tensor(reward).reshape(-1, 1).to(device)
        next_obs = torch.tensor(next_obs, requires_grad=False).to(device)
        done = torch.tensor(done).reshape(-1, 1).to(device)
        critic_loss, actor_loss = self.algo.learn(
            obs, action, reward, next_obs, done)  # 训练一次网络
        return critic_loss, actor_loss
