import torch
from torch.nn import MSELoss
from model import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# [r1,r2,...,rT] --> [G1,G2,...,GT]
def calc_reward_to_G(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # 从后往前推 最后一个的价值G就是其本身 前面的按照递推式更新:G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return reward_list


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

        baseline = reward.mean()
        G_value = calc_reward_to_G(reward)  # 每个动作对应的G不同
        log_prob = prob.log().gather(dim=1, index=action.long())  # 直接用这一行代码与下面5行等效

        # action_dim = prob.shape[-1]
        # # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        # action_onehot = self.one_hot(action, num_classes=action_dim)
        # prob = prob.log() * action_onehot
        # log_prob = prob.sum(axis=1).reshape(-1, 1)

        loss = torch.mean(-1 * log_prob * (G_value-baseline))  # 基线

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, path):
        self.model.load(path)
