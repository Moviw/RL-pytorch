from time import sleep
import numpy as np
import torch
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
        act = np.random.choice(prob.shape[-1], p=prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = torch.tensor(obs, dtype=torch.float32,
                           requires_grad=True).to(device)
        prob = self.algo.predict(obs)
        act = prob.cpu().detach().numpy().argmax()  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        """ 根据训练数据更新一次模型参数
        """
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        act = torch.tensor(act).int().reshape(-1, 1).to(device)
        reward = torch.tensor(reward).reshape(-1, 1).to(device)

        loss = self.algo.learn(obs, act, reward)
        return loss

    def save(self, save_path):
        self.algo.save(save_path)

    def load(self, path):
        self.algo.load(path)
