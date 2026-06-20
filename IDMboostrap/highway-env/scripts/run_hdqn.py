


import os
import random

import gym
import highway_env
import numpy as np
import torch
import torch.optim as optim

from Dscontroller import *
from agents.hdqn_mdp import HSAC
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule

#matplotlib.use('Agg')
TRAIN = True


def set_seed(seed: int) -> None:
    """Set all RNG seeds we can reach without changing training settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    SEEDS = [1, 2, 3]
    RUNS_PER_SEED = 1

    for seed in SEEDS:
        set_seed(seed)
        for run_idx in range(RUNS_PER_SEED):

            env = gym.make("highway-fast-v0")
            try:
                env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
            except Exception:
                pass
            obs = env.reset()# obs(5,5)
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
            batch_size=BATCH_SIZE,
            control_path=f"low_level_controller_seed{seed}.pth")
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
                stats_savepath=f"seed{seed}_run{run_idx}_hdqn.csv"
                np.savetxt(stats_savepath,stats,delimiter=",")

                print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))

