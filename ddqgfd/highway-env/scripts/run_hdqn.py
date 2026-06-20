import argparse
import random
import torch.optim as optim
import torch
from agents.hdqn_mdp import DDPGfD
from hdqn import ddpgfd_learning
from utils.schedule import LinearSchedule
import gym
import highway_env
import os
import numpy as np
from Dscontroller import *
#matplotlib.use('Agg')
TRAIN = True


def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        env.seed(seed)
    except Exception:
        pass
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=3000)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--model-dir", type=str, default="tmp")
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args()


def make_env():
    try:
        return gym.make("highway-fast-v0", disable_env_checker=True)
    except TypeError:
        return gym.make("highway-fast-v0")


if __name__ == '__main__':
    args = parse_args()

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


    seeds = [int(s) for s in args.seeds.split(",") if s]
    log_dir = args.model_dir
    os.makedirs(log_dir, exist_ok=True)
    for seed in seeds:

        env = make_env()
        obs = env.reset()# obs(5,5)
        #print(env.goal_reached(1))
        set_seed(seed, env)
        agent = DDPGfD(
        env=env,
        actor_lr=ACTOR_LEARNING_RATE,
        critic_lr=CRITIC_LEARNING_RATE,
        tau=TAU, 
        gamma=GAMMA, 
        device=DEVICE,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,)
        model=DSLaneChangeController()
        BETA=2
        #writer = SummaryWriter("C:\\Users\\Bruce Zhang\\Desktop\\newplotrecord\\trainhighlevel\\octobertest\\compareset\\set2\\log1")
        stats = None
        if TRAIN and not args.no_train:# skip visits 
            model_path = os.path.join(log_dir, "ddpgfd_seed{}.pth".format(seed))
            agent, stats,intrinsic_thousand_reward_totall = ddpgfd_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,
                model=model,
                save_path=model_path
            )
            stats_savepath=os.path.join(log_dir, "ddpgfd_seed{}_stats.csv".format(seed))
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
    
        if stats is not None:
            pass
