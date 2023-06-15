import time
import os
import torch
import gym
from algo import PPO
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

actor_lr = 1e-3
critic_lr = 1e-2
epochs = 10
GAMMA = 0.98
LMBDA = 0.95
EPSILON = 0.2
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_train_episode(env, agent):
    total_reward = 0
    state = env.reset()
    transition_dict = {'states': [], 'actions': [],
                       'next_states': [], 'rewards': [], 'dones': []}
    while True:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.update(transition_dict)
    return total_reward


def run_evaluate_episodes(agent, env, render=False):
    # 评估 agent, 跑 5 个episode，总reward求平均
    eval_reward = []
    for _ in range(5):
        obs = env.reset()
        total_reward = 0
        while True:
            # action = agent.predict(obs)
            action = agent.take_action(obs)
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
    model_save_path = relative_path+'\\PPO_model.pth'  # 模型参数保存地址
    log_save_path = f'{relative_path}\\logs\\{now}'  # 日志保存地址

    # logs目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(log_save_path)

    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPO(obs_dim, act_dim, actor_lr, critic_lr, LMBDA,
                epochs, EPSILON, GAMMA)

    # load model which already trained several times
    if os.path.exists(model_save_path):
        agent.load(model_save_path)

    max_episode = 500  # 训练的总episode数
    episode_per_evaluate = 50

    for episode in range(max_episode):

        episode += 1
        total_reward = run_train_episode(env, agent)
        writer.add_scalar('reward/train', total_reward, episode)

        if episode % episode_per_evaluate == 0:
            eval_reward = run_evaluate_episodes(agent, env, render=False)
            writer.add_scalar('reward/test', eval_reward,
                              episode/episode_per_evaluate)

            print('episode:%-4d |  Test reward:%.1f' %
                  (episode, eval_reward))

    # # save the parameters to ./model.pth
    # agent.save(model_save_path)
