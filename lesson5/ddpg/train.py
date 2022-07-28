from replay_memory import ReplayMemory
from time import sleep
from env import ContinuousCartPoleEnv
from algo import DDPG
from model import Net
from agent import Agent
import numpy as np
import gym
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 训练一个episode
def run_train_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.sample(batch_obs.astype('float32'))
        action = action[0]  # ContinuousCartPoleE输入的action为一个实数

        next_obs, reward, done, info = env.step(action)

        action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward

        if done or steps >= 200:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = action[0]  # ContinuousCartPoleE输入的action为一个实数

            steps += 1
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()
            if done or steps >= 200:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
MEMORY_SIZE = int(1e6)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE//20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 128
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.1  # 动作噪声方差
TRAIN_EPISODE = 3000  # 训练的总episode数

if __name__ == '__main__':
    env = ContinuousCartPoleEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 使用PARL框架创建agent
    model = Net(act_dim=act_dim, obs_dim=obs_dim).to(device)
    algorithm = DDPG(
        model, gamma=GAMMA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, act_dim, expl_noise=NOISE)

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)
    # 往经验池中预存数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    episode = 0
    for episode in range(TRAIN_EPISODE):
        total_reward = run_train_episode(agent, env, rpm)
        episode += 1

        if(episode+1) % 100 == 0:
            eval_reward = run_evaluate_episodes(agent, env, render=False)
            print('episode:%-4d |  Test reward:%.1f' %
                  (episode+1, eval_reward))
