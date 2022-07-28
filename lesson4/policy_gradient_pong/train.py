from algo import PolicyGradient
from model import FCN_model, CNN_model
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
        obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)
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
            obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float32()).reshape(1, 80, 80)


# [r1,r2,...,rT] --> [G1,G2,...,GT]
def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


if __name__ == '__main__':
    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    # run目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(f'run/CNN_{now}')
    env = gym.make('Pong-v0')
    obs_dim = 80 * 80
    act_dim = env.action_space.n

    # 根据parl框架构建agent
    # model = FCN_Model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    # 使用CNN处理图象
    model = CNN_model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    for i in range(1200, 3000):
        batch_obs, batch_action, batch_reward = run_train_episode(agent, env)
        if i % 10 == 0:
            print('episode:%3d  Train reward:%.1f' %
                  (i,  sum(batch_reward)))
        writer.add_scalar('reward/train', sum(batch_reward), i)

        batch_reward = calc_reward_to_go(batch_reward)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            print('Test reward: {}'.format(total_reward))
            writer.add_scalar('reward/test', total_reward, i)

        # save the parameters to ./model.ckpt
        if((i+1) % 300 == 0):
            agent.save(
                './policy_gradient_pong/CNN_model/{}_pong_model.ckpt'.format(i+1))
