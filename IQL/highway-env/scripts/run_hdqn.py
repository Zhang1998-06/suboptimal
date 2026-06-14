


import torch.optim as optim

import torch

from agents.hdqn_mdp import HCQL
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import os
import random
import numpy as np
from Dscontroller import *
#matplotlib.use('Agg')
TRAIN = True
if __name__ == '__main__':
    
    NUM_EPISODES = int(os.environ.get("NUM_EPISODES", 3000)) #  has to be integer multiple of mean number 
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




    seed_count = int(os.environ.get("NUM_SEEDS", 5))
    base_seed = int(os.environ.get("BASE_SEED", 0))
    print("Running training for {} episodes per seed.".format(NUM_EPISODES))
    for seed_index in range(seed_count):
        run_seed = base_seed + seed_index
        np.random.seed(run_seed)
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        env = gym.make("highway-fast-v0")
        try:
            env.reset(seed=run_seed)# obs(5,5)
        except TypeError:
            env.seed(run_seed)
            env.reset()
        #print(env.goal_reached(1))
        model_path = f"low_level_controller_seed{run_seed}.pth"
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
        control_path=model_path)
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
            stats_savepath=str(run_seed)+'hdqn.csv'
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
    
        print(f"Finished seed {run_seed}: saved controller to {model_path} and stats to {stats_savepath}")
        env.close()
