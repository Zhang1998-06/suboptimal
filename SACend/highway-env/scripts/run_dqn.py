import torch.optim as optim
from dqnmodel import SAC
from dqn import dqn_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import os
import torch

import numpy as np
#matplotlib.use('Agg')
TRAIN = True
if __name__ == '__main__':
    
    NUM_EPISODES = 3000 #  has to be integer multiple of mean number 
    BATCH_SIZE = 64


    GAMMA = 0.8
    REPLAY_MEMORY_SIZE = 50000
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-3
    ALPHA_LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    TAU = 0.005 
    TARGET_ENTROPY=-1
    for i in range(5):
        # forA2C should have use different learning rate
        env = gym.make("highway-fast-v0")
        obs = env.reset()# obs(5,5)
        agent = SAC(
        env=env,
        actor_lr=ACTOR_LEARNING_RATE,
        critic_lr=CRITIC_LEARNING_RATE,
        alpha_lr=ALPHA_LEARNING_RATE ,
        target_entropy=TARGET_ENTROPY, 
        tau=TAU, 
        gamma=GAMMA, 
        device=DEVICE,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,)
        if TRAIN:# skip visits 
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)
            agent, stats, = dqn_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,) 
            stats_savepath=str(i)+'end.csv'
            np.savetxt(stats_savepath,stats,delimiter=",")               


