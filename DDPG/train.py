import gym
import numpy as np
from algo import DDPG
from replay_memory import ReplayMemory
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter


MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
GAMMA = 0.98
TAU = 0.005  # 软更新参数
SIGMA = 0.01  # 高斯噪声标准差
actor_lr = 1e-2
critic_lr = 1e-2
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_train_episode(agent, env, rpm):
    # 训练一个episode
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.take_action(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            transition_dict = {'states': batch_obs, 'actions': batch_action,
                               'next_states': batch_next_obs, 'rewards': batch_reward, 'dones': batch_done}

            agent.update(transition_dict)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.take_action(obs)  # 预测动作，只选最优动作
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
    model_save_path = relative_path+'\\DDPG_model.pth'  # 模型参数保存地址
    log_save_path = f'{relative_path}\\logs\\{now}'  # 日志保存地址

    # logs目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(log_save_path)

    env = gym.make('Pendulum-v1')
    rpm = ReplayMemory(max_size=MEMORY_SIZE)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_up_bound = env.action_space.high[0]  # 动作最大值
    action_down_bound = env.action_space.low[0]  # 动作最大值
    agent = DDPG(state_dim,  action_dim, action_up_bound, action_down_bound,
                 SIGMA, actor_lr, critic_lr, TAU, GAMMA, device)

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
        total_reward = run_train_episode(agent, env, rpm)
        episode += 1  # 这里虽然自加1 但是不会影响外面for循环里episode的迭代

        writer.add_scalar('rewatd/train', total_reward, episode)

        if(episode % episode_per_evaluate == 0):
            # test part       render=True 查看显示效果
            eval_reward = run_evaluate_episodes(agent, env, render=True)
            writer.add_scalar('reward/test', eval_reward,
                              episode/episode_per_evaluate)

            print('episode:%-4d | Test reward:%.1f' % (
                episode,  eval_reward))

    # # save the parameters to ./model.pth
    # agent.save(model_save_path)
