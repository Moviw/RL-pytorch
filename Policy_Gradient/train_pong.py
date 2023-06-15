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
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
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


if __name__ == '__main__':
    now = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    relative_path = os.path.dirname(__file__)  # 获取相对路径 用来保存日志和模型参数
    model_save_path = relative_path+'\\PG_model.pth'  # 模型参数保存地址
    log_save_path = f'{relative_path}\\logs\\{now}'  # 日志保存地址

    # logs目录指向保存训练日志的总目录，后面还新加了一个根据当前时间设置的子目录，用于归类数据
    writer = SummaryWriter(log_save_path)

    env = gym.make('Pong-v0')
    obs_dim = 80 * 80
    act_dim = env.action_space.n

    # model = FCN_Model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    # 使用CNN处理图象
    model = CNN_model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

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
        if episode % episode_per_evaluate== 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            print('episode:%-4d | Test reward:%.1f' % (
                episode, total_reward))
            writer.add_scalar('reward/test', total_reward, episode)

        # # save the parameters to ./model.pth
        # if(episode % 300 == 0):
        #     agent.save(
        #         './policy_gradient_pong/CNN_model/{}_pong_model.pth'.format(i+1))

    # save the parameters to ./model.pth
    agent.save(model_save_path)