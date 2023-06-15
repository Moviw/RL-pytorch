import torch
from time import sleep
from replay_memory import ReplayMemory
from agent import Agent
from algo import DQN  # from parl.algorithms import DQN
import torch.nn as nn
from model import Net
import numpy as np
import os
import gym
import time
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0005  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等


# 训练一个episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    total_loss = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
            total_loss += train_loss

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, total_loss


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':
    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    relative_path = os.path.dirname(__file__)  # 获取相对路径 用来保存日志和模型参数
    model_save_path = relative_path+'\\DQN_model.pth'  # 模型参数保存地址
    log_save_path = f'{relative_path}\\logs\\{now}'  # 日志保存地址

    # logs目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(log_save_path)

    # CartPole-v0: expected reward > 180
    # MountainCar-v0 : expected reward > -120

    # env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)
    act_dim = env.action_space.n  # CartPole-v0: 2

    model = Net(obs_dim=obs_dim, act_dim=act_dim).to(device)
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim=act_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # load model which already trained several times
    # if os.path.exists(model_save_path):
    #     agent.load(model_save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    max_episode = 1000
    episode_per_evaluate = 100

    # start train
    for episode in range(max_episode):  # 训练max_episode个回合，test部分不计算入episode数量

        # train part
        total_reward, total_loss = run_train_episode(agent, env, rpm)
        episode += 1  # 这里虽然自加1 但是不会影响外面for循环里episode的迭代

        writer.add_scalar('reward/train', total_reward, episode)
        # writer.add_scalar('train/loss', total_loss, episode)

        if(episode % episode_per_evaluate == 0):
            # test part       render=True 查看显示效果
            eval_reward = run_evaluate_episodes(agent, env, render=False)
            writer.add_scalar('reward/test', eval_reward,
                              episode/episode_per_evaluate)

            print('episode:%-4d | e_greed:%.5f | Test reward:%.1f' % (
                episode, agent.e_greed, eval_reward))

    # save the parameters to ./model.pth
    agent.save(model_save_path)
