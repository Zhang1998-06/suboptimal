import numpy as np
from typing import List, Tuple, Dict
import gym
from gym.wrappers import RecordVideo, RecordEpisodeStatistics
import pickle
import csv
from Dscontroller import *
import numpy as np
from typing import List, Tuple, Dict
import gym
from gym.wrappers import RecordVideo,RecordEpisodeStatistics
#from utils import record_videos, show_videos
from tqdm.notebook import trange
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import highway_env
from Dscontroller import *
import numpy as np
from collections import defaultdict
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils.replay_memory import ReplayMemory
from utils import plotting
from Dscontroller import *

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
def get_action_index(action):
    actions = [
    (-1.0, -np.pi / 50),   # action 0
    (-1.0, 0.0),          # action 1
    (-1.0, np.pi / 50),   # action 2  rightlane
    (0.0, -np.pi / 50),   # action 3
    (0.0, 0.0),           # action 4
    (0.0, np.pi / 50),    # action 5   rightlane
    (1.0, -np.pi / 50),   # action 6 
    (1.0, 0.0),           # action 7
    (1.0, np.pi / 50)     # action 8   rightlane
        ]
    try:
        return actions.index(action)
    except ValueError:
        return "Action not found in the action space."    
    
# sarsa,done,ds
# state 
if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-fast-v0")
    model = DSLaneChangeController()
    trajectories = []

    # Open CSV file for writing
    with open("rule_based_trajectories.csv", "w", newline="") as csvfile:
        fieldnames = ["trajectory_index","time_steps","state", "action", "next_state", "reward", "done","ds_value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1000):  # Adjust range for multiple episodes
            accu_reward = 0
            reward = 0
            done = False
            current_state, rule_state = env.reset()
            env.set_goal("IDLE")
            time_i = 0

            while not done:
                # Ensure states are flattened
                current_state = current_state.reshape(-1)
                rule_state = rule_state.reshape(-1)

                # Create joint state-goal representation
                joint_state_goal = np.concatenate([current_state, [env.lat_center()], [env.vehicle.on_road]])

                # Predict using the rule-based controller
                target_speed, target_lane, RL_weight = model.control(True, env, rule_state)
                goal_1 = interperate_ds_command(env, target_speed, target_lane)
                env.set_goal(goal_1)
                low_level_index=get_action_index(env.goal_low_action())
                # Step the environment
                next_state, high_level_reward, done, info = env.step(low_level_index)
                if high_level_reward>-10:
                    reward_min = 0 - 0.5*2#*RL_weight
                    reward_max = 1
                    high_level_reward = (high_level_reward - reward_min) / (reward_max - reward_min)
                rule_next_state = info["rule_state"]
                next_state = next_state.reshape(-1)
                joint_next_state_goal = np.concatenate([next_state, [env.lat_center()], [env.vehicle.on_road]])

                # Validate and prepare data for storage
                trajectory = {
                    "trajectory_index":i,
                    "time_steps":time_i,
                    "state": joint_state_goal.tolist(),
                    "action": low_level_index,
                    "next_state": joint_next_state_goal.tolist(),
                    "reward": float(high_level_reward),  # Ensure it's a float
                    "done": bool(done),
                    "ds_value":RL_weight
                }
                trajectories.append(trajectory)

                # Write to CSV (ensure data consistency)
                writer.writerow({
                    "trajectory_index":i,
                    "time_steps":time_i,
                    "state": joint_state_goal.tolist(),
                    "action": low_level_index,
                    "next_state": joint_next_state_goal.tolist(),
                    "reward": float(high_level_reward),
                    "done": bool(done),
                    "ds_value":RL_weight
                })

                # Update states
                current_state = next_state
                rule_state = rule_next_state
                accu_reward += high_level_reward
                time_i += 1

    # Save trajectories to a .pkl file
    with open("rule_based_trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    env.close()