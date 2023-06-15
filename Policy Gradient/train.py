from algo import PolicyGradient
from model import Model
from agent import Agent
import numpy as np
import os
import gym
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LEARNING_RATE = 1e-3


# 训练一个episode
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return np.array(obs_list), np.array(action_list), np.array(reward_list)


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    relative_path = os.path.dirname(__file__)  # 获取相对路径 用来保存日志和模型参数
    model_save_path = relative_path+'\\PG_model.pth'  # 模型参数保存地址
    log_save_path = f'{relative_path}\\logs\\{now}'  # 日志保存地址

    # logs目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(log_save_path)

    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    algorithm = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(algorithm)

    # load model which already trained several times
    if os.path.exists(model_save_path):
        agent.load(model_save_path)

    max_episode = 1000
    episode_per_evaluate = 100

    # start train
    for episode in range(max_episode):
        batch_obs, batch_action, batch_reward = run_train_episode(agent, env)

        episode += 1  # 这里虽然自加1 但是不会影响外面for循环里episode的迭代
        writer.add_scalar('reward/train', sum(batch_reward), episode)

        agent.learn(batch_obs, batch_action, batch_reward)
        if episode % episode_per_evaluate == 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            print('episode:%-4d | Test reward:%.1f' % (
                episode, total_reward))
            writer.add_scalar('reward/test', total_reward, episode)

    # save the parameters to ./model.pth
    agent.save(model_save_path)


if __name__ == '__main__':
    main()
