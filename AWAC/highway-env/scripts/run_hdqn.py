# Ensure restricted environments don't require shared memory for BLAS/OpenMP
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.optim as optim

import shutil

from agents.hdqn_mdp import HCQL
from hdqn import hdqn_learning
from utils.schedule import LinearSchedule
from utils.plotting import EpisodeStats
import gym
import highway_env
import numpy as np
import random
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




    BASE_SEED = 0
    TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", NUM_EPISODES))
    EPISODES_PER_RUN = int(os.environ.get("EPISODES_PER_RUN", TOTAL_EPISODES))
    if EPISODES_PER_RUN <= 0:
        raise ValueError("EPISODES_PER_RUN must be a positive integer")
    if TOTAL_EPISODES % 5 != 0:
        raise ValueError("TOTAL_EPISODES must be divisible by the stats check frequency (5).")
    START_SEED = int(os.environ.get("START_SEED", 0))
    END_SEED = int(os.environ.get("END_SEED", 4))
    if not (0 <= START_SEED <= END_SEED <= 4):
        raise ValueError("START_SEED and END_SEED must satisfy 0 <= START_SEED <= END_SEED <= 4")
    progress_dir = os.path.join("tmp", "awac_progress")
    os.makedirs(progress_dir, exist_ok=True)

    for i in range(START_SEED, END_SEED + 1):

        seed = BASE_SEED + i
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = gym.make("highway-fast-v0")
        if hasattr(env, "seed"):
            env.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        obs = env.reset()# obs(5,5)

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
        batch_size=BATCH_SIZE,)

        seed_control_path = f"low_level_controller_seed_{i}.pth"
        agent.control_path = seed_control_path

        stats_savepath=str(i)+'hdqn.csv'
        plot_prefix=str(i)+'dqn'
        progress_path = os.path.join(progress_dir, f"seed_{i}_progress.npz")

        stats_buffers = {field: np.array([], dtype=float) for field in EpisodeStats._fields}
        intrinsic_history = np.array([], dtype=float)
        completed_episodes = 0
        finished_training = False

        if os.path.exists(stats_savepath):
            stats_matrix = np.loadtxt(stats_savepath, delimiter=",")
            if stats_matrix.ndim == 1:
                stats_matrix = stats_matrix[:, np.newaxis]
            for idx, field in enumerate(EpisodeStats._fields):
                if idx < stats_matrix.shape[0]:
                    stats_buffers[field] = stats_matrix[idx]
            completed_episodes = stats_matrix.shape[1]
            if completed_episodes >= TOTAL_EPISODES:
                finished_training = True
        if os.path.exists(progress_path):
            data = np.load(progress_path, allow_pickle=True)
            completed_episodes = int(data["completed"])
            for field in EpisodeStats._fields:
                stats_buffers[field] = data[field]
            intrinsic_history = data["intrinsic"]
            print(f"Loaded progress for seed {i}: {completed_episodes}/{TOTAL_EPISODES} episodes.")
            if os.path.exists(seed_control_path):
                agent.load()
        elif os.path.exists(seed_control_path):
            agent.load()

        if TRAIN:
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)

            ran_chunk = False
            if completed_episodes < TOTAL_EPISODES:
                remaining = TOTAL_EPISODES - completed_episodes
                episodes_this_run = min(remaining, EPISODES_PER_RUN)
                if episodes_this_run % 5 != 0 and remaining >= 5:
                    episodes_this_run -= episodes_this_run % 5
                if episodes_this_run == 0:
                    episodes_this_run = remaining
                print(f"Training seed {i}: episodes {completed_episodes} -> {completed_episodes + episodes_this_run}")
                agent, stats_chunk,intrinsic_chunk = hdqn_learning(
                    env=env,
                    agent=agent,
                    num_episodes=episodes_this_run,
                )
                for field in EpisodeStats._fields:
                    stats_buffers[field] = np.concatenate([stats_buffers[field], getattr(stats_chunk, field)])
                intrinsic_history = np.concatenate([intrinsic_history, np.array(intrinsic_chunk, dtype=float)])
                completed_episodes += episodes_this_run
                np.savez(progress_path,
                         completed=completed_episodes,
                         intrinsic=intrinsic_history,
                         **stats_buffers)
                print(f"Progress saved for seed {i}: {completed_episodes}/{TOTAL_EPISODES}")
                ran_chunk = True

            if completed_episodes >= TOTAL_EPISODES:
                stats_matrix = np.stack([stats_buffers[field] for field in EpisodeStats._fields])
                np.savetxt(stats_savepath, stats_matrix, delimiter=",")
                stats_final = EpisodeStats(*[stats_buffers[field] for field in EpisodeStats._fields])

                print("intrinsic_thousand_reward_totall{}".format(intrinsic_history))


                if os.path.exists(progress_path):
                    os.remove(progress_path)

                if os.path.exists(seed_control_path):
                    shutil.copy(seed_control_path, 'low_level_controller.pth')
                finished_training = True
                print(f"Seed {i} training complete.")
            elif ran_chunk:
                print(f"Seed {i}: {completed_episodes}/{TOTAL_EPISODES} episodes finished. Re-run to continue.")

        env.close()
