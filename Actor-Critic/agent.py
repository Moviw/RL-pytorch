import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, algorithm):
        self.algo = algorithm

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        prob = self.algo.predict(obs)
        prob = prob.cpu().detach().numpy()
        action = np.random.choice(prob.shape[-1], p=prob)  # 根据动作概率选取动作
        return action

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        predict_q = self.algo.predict(obs)
        act = predict_q.cpu().detach().numpy().argmax()
        return act

    def learn(self, obs, action, reward, next_obs, done):
        """ 根据训练数据更新一次模型参数
        """
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        action = torch.tensor(
            action).reshape(-1, 1).to(device)
        reward = torch.tensor(
            reward, dtype=torch.float32).reshape(-1, 1).to(device)
        next_obs = torch.tensor(
            next_obs, requires_grad=False, dtype=torch.float32).to(device)
        done = torch.tensor(
            done, dtype=torch.float32).reshape(-1, 1).to(device)
        critic_loss, actor_loss = self.algo.learn(
            obs, action, reward, next_obs, done)  # 训练一次网络
        return critic_loss, actor_loss

    def save(self, save_path):
        self.algo.save(save_path)

    def load(self, path):
        self.algo.load(path)
