import argparse
import os
os.environ["KMP_CREATE_SHM"] = "0"
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass



import torch.optim as optim

from agents.hdqn_mdp import HCQL
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import numpy as np
from Dscontroller import *
import random
#matplotlib.use('Agg')
TRAIN = True
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train TD3+BC controllers for multiple seeds.")
    parser.add_argument("--seeds", type=int, nargs="*", help="Optional explicit list of seeds to train.")
    args = parser.parse_args()
    
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
    BASE_SEED = 0

    default_seeds = [BASE_SEED + i for i in range(5)]
    seed_list = args.seeds if args.seeds else default_seeds

    for seed in seed_list:
        controller_path = f"low_level_controller_seed_{seed}.pth"
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        env = gym.make("highway-fast-v0")
        try:
            env.seed(seed)
        except AttributeError:
            pass
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        obs = env.reset()# obs(5,5)
        #print(env.goal_reached(1))
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
        control_path=controller_path)
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
            stats_savepath=f'hdqn_seed_{seed}.csv'
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
    
