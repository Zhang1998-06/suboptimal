import argparse
import torch
import torch.optim as optim

from agents.hdqn_mdp import DQfDAgent
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
    parser = argparse.ArgumentParser(description="Train DQfD agents.")
    parser.add_argument("--num-episodes", type=int, default=3000)
    parser.add_argument("--num-seeds", type=int, default=5)
    args = parser.parse_args()

    NUM_EPISODES = args.num_episodes #  has to be integer multiple of mean number 
    BATCH_SIZE = 64


    GAMMA = 0.8
    REPLAY_MEMORY_SIZE = 50000
    CRITIC_LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    TAU = 0.005 
    DEMO_MARGIN = 0.8
    DEMO_WEIGHT = 1.0
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_STEPS = 100000

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for i in range(args.num_seeds):
        seed = i
        seed_everything(seed)

        env = gym.make("highway-fast-v0")
        try:
            env.seed(seed)
            env.action_space.seed(seed)
        except AttributeError:
            pass
        obs = env.reset()# obs(5,5)
        #print(env.goal_reached(1))
        agent = DQfDAgent(
            env=env,
            q_lr=CRITIC_LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            device=DEVICE,
            replay_memory_size=REPLAY_MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            demo_margin=DEMO_MARGIN,
            demo_weight=DEMO_WEIGHT,
            model_path=f"dqfd_seed_{seed}.pth",
        )
        #agent.load()
        model=DSLaneChangeController()
        BETA=2
        #writer = SummaryWriter("C:\\Users\\Bruce Zhang\\Desktop\\newplotrecord\\trainhighlevel\\octobertest\\compareset\\set2\\log1")
        if TRAIN:# skip visits 
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)
            epsilon_schedule = LinearSchedule(EPSILON_DECAY_STEPS, EPSILON_END, EPSILON_START)
            agent, stats,intrinsic_thousand_reward_totall = hdqn_learning(
                env=env,
                agent=agent,
                num_episodes=NUM_EPISODES,
                BETA=BETA,
                model=model,
                epsilon_schedule=epsilon_schedule,
            )
            stats_savepath=str(i)+'hdqn.csv'
            np.savetxt(stats_savepath,stats,delimiter=",")
     
            print("intrinsic_thousand_reward_totall{}".format(intrinsic_thousand_reward_totall))
    
