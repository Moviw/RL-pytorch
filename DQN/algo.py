import copy
import torch
from torch.nn import MSELoss
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN():
    def __init__(self, model, gamma=None, lr=None):
        """ 
        DQN algorithm

         Args:
             model (parl.Model): 定义Q函数的前向网络结构
             gamma (float): reward的衰减因子
             lr (float): learning_rate 学习率.
         """
        # checks
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.requires_grad = False  # 这条居然不写也行 不知道具体原因是什么

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, obs):
        """ 使用self.model的网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model(obs)

    def one_hot(self, action, num_classes):
        action = action.T
        assert isinstance(num_classes, int)
        action_onehot = np.zeros([action.shape[-1], num_classes])
        for i in range(action.shape[-1]):
            action_onehot[i, action[0, i]] = 1
        return torch.tensor(action_onehot,  requires_grad=True).to(device)

    def learn(self, obs, action, reward, next_obs, done):
        """ 使用DQN算法更新self.model的value网络
        """
        # 获取Q(s,a)
        q_values = self.model(obs).gather(
            dim=1, index=action.long())  # 直接用这一行代码与下面12行等效

        # q_values = self.model(obs)

        # action_dim = q_values.shape[-1]
        # # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        # action_onehot = self.one_hot(action, num_classes=action_dim)

        # # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # # 比如：q_values = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        # #  ==> q_values = [[3.9]]
        # q_values = q_values * action_onehot
        # q_values = q_values.sum(axis=1).reshape(-1, 1)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        with torch.no_grad():
            next_q_values = self.target_model(next_obs).cpu().detach().numpy()
            next_q_values = torch.Tensor(
                next_q_values.max(axis=1)).reshape(-1, 1).to(device)
            temp = reward + self.gamma * next_q_values * (1 - done)
            expected_q_values = temp.clone().detach().requires_grad_(True)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        loss = self.mse_loss(q_values.float(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, path):
        self.model.load(path)
