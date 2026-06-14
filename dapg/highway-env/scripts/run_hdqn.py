import argparse

import torch.optim as optim

import torch

from agents.hdqn_mdp import HSAC
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import os
import numpy as np
import random
from Dscontroller import *
#matplotlib.use('Agg')
TRAIN = True
if __name__ == '__main__':
    def make_env():
        try:
            return gym.make("highway-fast-v0", disable_env_checker=True)
        except TypeError:
            return gym.make("highway-fast-v0")
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def parse_seeds(seeds_value):
        return [int(seed.strip()) for seed in seeds_value.split(",") if seed.strip()]

    parser = argparse.ArgumentParser(description="Train HDQN baseline with multiple seeds.")
    parser.add_argument("--method", choices=["suboptimal", "dapg"], default="dapg")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num-episodes", type=int, default=3000)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    NUM_EPISODES = args.num_episodes #  has to be integer multiple of mean number 
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




    seeds = parse_seeds(args.seeds)
    os.makedirs(args.models_dir, exist_ok=True)
    for seed in seeds:
        set_seed(seed)
        env = make_env()
        try:
            obs = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            obs = env.reset()
        env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        #print(env.goal_reached(1))
        model_path = os.path.join(args.models_dir, f"{args.method}_seed{seed}.pth")
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
        control_path=model_path,)
        #agent.load()
        model=DSLaneChangeController()
        BETA=2
        output_dir = os.path.join(args.output_dir, args.method, f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)
        #writer = SummaryWriter("C:\\Users\\Bruce Zhang\\Desktop\\newplotrecord\\trainhighlevel\\octobertest\\compareset\\set2\\log1")
        if TRAIN:# skip visits 
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)
            agent, stats,intrinsic_thousand_reward_totall = hdqn_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,
                BETA=BETA,
                model=model,
                method=args.method
            )
            stats_savepath=os.path.join(output_dir, "hdqn.csv")
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
    
