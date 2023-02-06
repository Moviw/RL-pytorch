import torch
import torch.nn.functional as F


class PPO_Net(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO_Net, self).__init__()
        self.actor_model = Actor(state_dim, action_dim)
        self.critic_model = Critic(state_dim)

    def act(self, obs):
        return self.actor_model(obs)

    def criticize(self, obs):
        return self.critic_model(obs)

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        hidden_dim = 256
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        hidden_dim = 256
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
