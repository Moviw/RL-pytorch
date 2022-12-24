import torch
from torch.nn import MSELoss
from model import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyGradient():
    def __init__(self, model, lr):
        """ Policy Gradient algorithm

        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """
        assert isinstance(lr, float)

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def one_hot(self, action, num_classes):
        action = action.T
        assert isinstance(num_classes, int)
        action_onehot = np.zeros([action.shape[-1], num_classes])
        for i in range(action.shape[-1]):
            action_onehot[i, action[0, i]] = 1
        return torch.tensor(action_onehot,  requires_grad=True).to(device)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        prob = self.model(obs)  # 获取输出动作概率

        log_prob = prob.log().gather(dim=1, index=action.long())  # 直接用这一行代码与下面5行等效

        # action_dim = prob.shape[-1]
        # # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        # action_onehot = self.one_hot(action, num_classes=action_dim)
        # prob = prob.log() * action_onehot
        # log_prob = prob.sum(axis=1).reshape(-1, 1)

        reward = reward.reshape(-1, 1)
        loss = torch.mean(-1 * log_prob * reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, path):
        self.model.load(path)
