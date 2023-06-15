from time import sleep
from algo import A2C
from model import Net
from agent import Agent
import numpy as np
import gym
import torch
import os
import time
# from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子


def run_train_episode(agent, env):
    # 训练一个episode
    obs = env.reset()
    total_reward = 0
    batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = [
    ], [], [], [], []  # 每个step创建一个空数组存储当前元素，这样方便在后面转化为Tensor
    while True:
        action = agent.sample(obs)
        next_obs, reward, done, info = env.step(action)

        batch_obs.append(obs)
        batch_action.append(action)
        batch_reward.append(reward)
        batch_next_obs.append(next_obs)
        batch_done.append(done)

        obs = next_obs
        total_reward += reward

        if done:
            break

    agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                batch_done)

    return total_reward


def run_evaluate_episodes(agent, env, render=False):
    # 评估 agent, 跑 5 个episode，总reward求平均
    eval_reward = []
    for _ in range(5):
        obs = env.reset()
        total_reward = 0
        while True:
            action = agent.predict(obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            obs = next_obs

            if render:
                env.render()
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':
    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    relative_path = os.path.dirname(__file__)  # 获取相对路径 用来保存日志和模型参数
    model_save_path = relative_path+'\\A2C_model.pth'  # 模型参数保存地址
    log_save_path = f'{relative_path}\\logs\\{now}'  # 日志保存地址

    # logs目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    # writer = SummaryWriter(log_save_path)

    env = gym.make("CartPole-v0")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 使用PARL框架创建agent
    model = Net(obs_dim=obs_dim, act_dim=act_dim).to(device)
    algorithm = A2C(
        model, gamma=GAMMA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm)

    # load model which already trained several times
    # if os.path.exists(model_save_path):
    #     agent.load(model_save_path)

    max_episode = 1000  # 训练的总episode数
    episode_per_evaluate = 100

    for episode in range(max_episode):

        total_reward = run_train_episode(agent, env)
        # writer.add_scalar('reward/train', total_reward, episode)
        episode += 1

        if episode % episode_per_evaluate == 0:
            eval_reward = run_evaluate_episodes(agent, env, render=False)
            # writer.add_scalar('reward/test', eval_reward,
            #                   episode/episode_per_evaluate)

            print('episode:%-4d |  Test reward:%.1f' %
                  (episode, eval_reward))

    # save the parameters to ./model.pth
    # agent.save(model_save_path)
