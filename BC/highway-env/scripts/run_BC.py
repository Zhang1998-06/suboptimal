# main.py

import numpy as np
import gym
import highway_env  # remove if not using this environment
import os
import pickle

# Import BC classes/functions
from bc_agent import BehaviorClone, bc_learning

if __name__ == "__main__":
    # 1) Create environment (adapt name or settings as needed)
    env = gym.make("highway-fast-v0")

    # Suppose your environment observation is (5, 5) => flatten to 25
    # Adjust 'state_dim' as needed.
    state_dim = 27
    hidden_dim = 256
    action_dim = 9  # same discrete action space as your original code

    # 2) Load offline (expert) dataset from pickle
    # The pickle has a list of dicts: [ {"state":..., "action":..., ...}, ... ]
    with open("rule_based_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)

    # Build arrays for states and actions
    all_states = []
    all_actions = []
    for t in trajectories:
        # Flatten state if it's 2D (e.g. (5,5))
        s = np.array(t["state"], dtype=np.float32)
        if len(s.shape) > 1:
            s = s.flatten()  # now shape (25,)

        a = t["action"]  # integer in [0..8], hopefully

        all_states.append(s)
        all_actions.append(a)

    expert_s = np.array(all_states, dtype=np.float32)   # shape (N, 25)
    expert_a = np.array(all_actions, dtype=np.int64)    # shape (N,)

    print("Loaded offline dataset:")
    print("expert_s shape:", expert_s.shape)
    print("expert_a shape:", expert_a.shape)

    # 3) Create BC agent
    bc_agent = BehaviorClone(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        lr=1e-3
    )

    # 4) Train BC (offline + environment rollouts)
    NUM_EPISODES = 2000
    BATCH_SIZE = 64
    CHECK_FREQUENCY = 5

    bc_agent, stats = bc_learning(
        env=env,
        bc_agent=bc_agent,
        expert_s=expert_s,
        expert_a=expert_a,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        check_frequence=CHECK_FREQUENCY
    )

    # 5) Save stats to CSV
    np.savetxt(
        "bc_stats.csv",
        np.column_stack([
            stats.episode_lengths,
            stats.episode_rewards,
            stats.crash_times,
            stats.trap,
            stats.average_speed,
            stats.distance,
            stats.episode_low_level_reward,
            stats.goal_reached_times,
            stats.episode_min_headway,
            stats.episode_min_ttc,
            stats.episode_max_lateral_jerk,
            stats.episode_lane_invasion,
        ]),
        delimiter=",",
        header=(
            "length,reward,crash,trap,avg_speed,distance,low_level_reward,goal_reached,"
            "min_headway,min_ttc,max_lateral_jerk,lane_invasion"
        ),
        comments=""
    )

    # 6) Plot results


