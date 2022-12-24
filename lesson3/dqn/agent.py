import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        assert isinstance(act_dim, int)
        self.algo = algorithm
        self.act_dim = act_dim

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

        self.target_step = 0  # 当前进行的training step数
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        sample = np.random.random()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.choice(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = torch.Tensor(obs).to(device)
        predict_q = self.algo.predict(obs)
        act = torch.argmax(predict_q).cpu().numpy()  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, done):
        """ 根据训练数据更新一次模型参数
        """
        if (self.target_step % self.update_target_steps == 0):
            self.algo.sync_target()
        self.target_step += 1

        obs = torch.tensor(obs, requires_grad=True).to(device)
        act = torch.tensor(act).int().reshape(-1, 1).to(device)
        reward = torch.tensor(reward).reshape(-1, 1).to(device)
        next_obs = torch.tensor(next_obs).to(device)
        done = torch.tensor(done).reshape(-1, 1).to(device)
        loss = self.algo.learn(obs, act, reward, next_obs, done)  # 训练一次网络
        return loss

    def save(self, save_path):
        self.algo.save(save_path)

    def load(self, path):
        self.algo.load(path)
