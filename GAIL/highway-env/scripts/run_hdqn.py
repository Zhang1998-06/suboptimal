import torch.optim as optim
import torch
import gym
import highway_env
import os
import numpy as np

from agents.hdqn_mdp import HSAC  # or your HSAC definition
from hdqn import hdqn_learning    # your updated python file 2
from utils.schedule import LinearSchedule
# from Dscontroller import * # If you need more logic from Dscontroller

TRAIN = True

if __name__ == '__main__':
    NUM_EPISODES = 3000
    BATCH_SIZE = 64
    GAMMA = 0.8
    REPLAY_MEMORY_SIZE = 50000
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-3
    ALPHA_LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    TAU = 0.005
    TARGET_ENTROPY = -1

    # Possibly repeat runs
    for i in range(5):
        env = gym.make("highway-fast-v0")
        obs = env.reset()

        agent = HSAC(
            env=env,
            actor_lr=ACTOR_LEARNING_RATE,
            critic_lr=CRITIC_LEARNING_RATE,
            alpha_lr=ALPHA_LEARNING_RATE,
            target_entropy=TARGET_ENTROPY,
            tau=TAU,
            gamma=GAMMA,
            device=DEVICE,
            replay_memory_size=REPLAY_MEMORY_SIZE,
            batch_size=BATCH_SIZE,
        )

        if TRAIN:
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)

            # 'model' argument is unused in your example, but you can pass None or something
            agent, stats, intrinsic_thousand_reward_totall = hdqn_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,
            )

            stats_savepath = str(i) + '_hdqn.csv'
            np.savetxt(stats_savepath, stats, delimiter=",")

            print("intrinsic_thousand_reward_totall", intrinsic_thousand_reward_totall)
