


import torch.optim as optim

import torch

import random
from agents.hdqn_mdp import HSAC
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import os
import numpy as np
from Dscontroller import *
#matplotlib.use('Agg')
TRAIN = os.getenv("TRAIN", "1") != "0"


def make_env():
    try:
        return gym.make("highway-fast-v0", disable_env_checker=True)
    except TypeError as exc:
        if "disable_env_checker" not in str(exc):
            raise
        return gym.make("highway-fast-v0")

if __name__ == '__main__':
    
    NUM_EPISODES = int(os.getenv("NUM_EPISODES", "3000")) #  has to be integer multiple of mean number 
    NUM_SEEDS = int(os.getenv("NUM_SEEDS", "5"))
    BASE_SEED = int(os.getenv("BASE_SEED", "0"))
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




    for i in range(NUM_SEEDS):
        seed = BASE_SEED + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = make_env()
        if hasattr(env, "seed"):
            env.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        obs = env.reset()# obs(5,5)
        #print(env.goal_reached(1))
        agent = HSAC(
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
        agent.control_path = "low_level_controller_seed{}.pth".format(seed)
        #agent.load()
        model=DSLaneChangeController()
        BETA=2
        #writer = SummaryWriter("C:\\Users\\Bruce Zhang\\Desktop\\newplotrecord\\trainhighlevel\\octobertest\\compareset\\set2\\log1")
        if TRAIN:# skip visits 
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)
            agent, stats,intrinsic_thousand_reward_totall = hdqn_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,
                BETA=BETA,
                model=model
            )
            stats_savepath=str(seed)+'hdqn.csv'
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
    
