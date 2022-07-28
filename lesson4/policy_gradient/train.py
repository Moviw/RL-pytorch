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
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# [r1,r2,...,rT] --> [G1,G2,...,GT]
def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return reward_list


def main():
    env = gym.make('CartPole-v0')
    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    # run目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(f'run/{now}')
    # env = env.unwrapped # Cancel the minimum score limit
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    # 加载模型并评估
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    for i in range(1000):
        batch_obs, batch_action, batch_reward = run_train_episode(agent, env)
        if i % 10 == 0:
            print('episode:%3d | Train reward:%.1f' % (
                i,  sum(batch_reward)))
        writer.add_scalar('reward/train', sum(batch_reward), i)
        batch_reward = calc_reward_to_go(batch_reward)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            print('Test reward: {}'.format(total_reward))
            writer.add_scalar('reward/test', total_reward, i)

    # save the parameters to ./model.ckpt
    # agent.save('./model.ckpt')


if __name__ == '__main__':
    main()
