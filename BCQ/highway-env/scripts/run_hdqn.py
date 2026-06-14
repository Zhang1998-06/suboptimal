import random

import torch.optim as optim

import torch

from agents.hdqn_mdp import HCQL
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import os
import numpy as np
from Dscontroller import *
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
    NUM_SEEDS = 5
    BASE_SEED = 42
    CHECKPOINT_DIR = "models"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for i in range(NUM_SEEDS):
        run_seed = BASE_SEED + i
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        env = gym.make("highway-fast-v0")
        if hasattr(env, "seed"):
            env.seed(run_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(run_seed)
        obs = env.reset()# obs(5,5)
        #print(env.goal_reached(1))
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"low_level_controller_seed_{i}.pth")

        agent = HCQL(
        env=env,
        actor_lr=ACTOR_LEARNING_RATE,
        critic_lr=CRITIC_LEARNING_RATE,
        alpha_lr=ALPHA_LEARNING_RATE ,
        target_entropy=TARGET_ENTROPY, 
        tau=TAU, 
        gamma=GAMMA, 
        device=DEVICE,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        control_path=checkpoint_path,
        )
        #agent.load()

        #writer = SummaryWriter("C:\\Users\\Bruce Zhang\\Desktop\\newplotrecord\\trainhighlevel\\octobertest\\compareset\\set2\\log1")
        if TRAIN:# skip visits 
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)
            agent, stats,intrinsic_thousand_reward_totall = hdqn_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,
            )
            stats_savepath=str(i)+'hdqn.csv'
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
            print(f"Saved controller for seed {i} at {checkpoint_path}")
    
