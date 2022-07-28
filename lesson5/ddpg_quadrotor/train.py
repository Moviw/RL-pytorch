from rlschool import make_env
from replay_memory import ReplayMemory
from time import sleep
from algo import DDPG
from model import Net
from agent import Agent
import numpy as np
import gym
import torch
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActionMappingWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]
        assert self.high_bound > self.low_bound

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act):
        assert np.all(((model_output_act <= 1.0 + 1e-3), (model_output_act >= -1.0 - 1e-3))), \
            'the action should be in range [-1, 1] !'
        assert self.high_bound > self.low_bound
        mapped_action = self.low_bound + \
            (model_output_act - (-1.0)) * \
            ((self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)
        return self.env.step(mapped_action)

    def render(self):
        self.env.render()


# 训练一个episode
def run_train_episode(agent, env, rpm):
    obs = env.reset()
    episode_steps, episode_reward = 0, 0
    done = False
    while not done:
        episode_steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.sample(batch_obs.astype('float32'))
        action = action[0]

        next_obs, reward, done, info = env.step(action)

        action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (episode_steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        episode_reward += reward

    return episode_reward, episode_steps


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, eval_episodes, render=True):
    eval_reward = []
    for i in range(eval_episodes):
        total_reward = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


ACTOR_LR = 0.0002  # Actor网络的 learning rate
CRITIC_LR = 0.001  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
MEMORY_SIZE = int(25000)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE//20  # 预存一部分经验之后再开始训练
REWARD_SCALE = 0.01  # reward 缩放系数

BATCH_SIZE = 256
EVAL_EPISODES = 5
NOISE = 0.1  # 动作噪声方差

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_total_steps",
        default=2000000,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='The step interval between two consecutive evaluations')
    args = parser.parse_args()

    env = make_env('Quadrotor', task='hovering_control')
    env = ActionMappingWrapper(env)
    env.reset()

    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    # run目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(f'./ddpg_quadrotor/run/{now}')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 使用PARL框架创建agent
    model = Net(act_dim=act_dim, obs_dim=obs_dim).to(device)
    # model = torch.load('ddpg_quadrotor/model/model.pt')
    algorithm = DDPG(
        model, gamma=GAMMA,  actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, act_dim, expl_noise=NOISE)

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)
    # 往经验池中预存数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    test_flag, total_steps = 0, 0
    while total_steps < args.train_total_steps:
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps
        writer.add_scalar('train/episode_reward', episode_reward, total_steps)
        print('Total Steps: {} Reward: {}'.format(total_steps, episode_reward))

        # Evaluate episode
        if total_steps // args.test_every_steps > test_flag:
            test_flag = total_steps // args.test_every_steps
            avg_reward = run_evaluate_episodes(
                agent, env, EVAL_EPISODES)
            writer.add_scalar('test/episode_reward',
                              avg_reward, total_steps)
            print('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))

    torch.save(model, 'ddpg_quadrotor/model/model.pt')
