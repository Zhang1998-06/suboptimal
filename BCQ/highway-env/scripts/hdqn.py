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
    
def stochastic_multi_actions(action_indices, N=9, theta=0.2):
    # If all actions are selected, return a uniform distribution
    if len(action_indices) == N:
        dist = torch.full((N,), 1.0 / N)
    else:
        # Start with a distribution where theta is evenly spread across non-selected actions
        dist = torch.full((N,), theta / (N - len(action_indices)))
        
        # Assign a higher probability to each action in action_indices
        for action in action_indices:
            dist[action] = 1 - theta / len(action_indices)
        
        # Normalize the distribution to ensure it sums to 1
        dist /= dist.sum()
    
    return dist


def hdqn_learning(
    env,
    agent,
    num_episodes,

    ):

    """The h-DQN learning algorithm. load high-level controller, train low-level controller
       the high level controller give the low level controller a prefered goal
       in the set, low level use DQN 
       in other set low level use HDQNstats.episode_low_level_reward
       two set of reward 
    """
    from utils.replay_memory import ReplayMemory
    offline_replay = ReplayMemory(capacity=50000)


    # -----------------------------------
    # NEW CODE (B): Load from PKL (or CSV) into offline_replay
    # -----------------------------------
    import pickle
    with open("rule_based_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    for t in trajectories:
        state      = t["state"]
        action     = t["action"]
        next_state = t["next_state"]
        reward     = t["reward"]
        done       = t["done"]
        Ds_value   = t["ds_value"]
        # Push them into the OFFLINE replay
        offline_replay.push(state, action, next_state, reward, done)
    check_frequence=5
    beta_start = 1.0  # Initial value of beta
    beta_end = 0.01    # Final value of beta
    annealing_episodes = 500 
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        crash_times=np.zeros(num_episodes),
        trap=np.zeros(num_episodes),
        average_speed=np.zeros(num_episodes),
        distance=np.zeros(num_episodes),
        episode_low_level_reward=np.zeros(num_episodes),
        goal_reached_times=np.zeros(num_episodes),
        )
    n_thousand_episode = int(np.floor(num_episodes / check_frequence))
    total_timestep = 0
    overall_episode=0
    intrinsic_thousand_reward_totall=[]
    extrinsic_thousand_reward_totall=[]
    best_episode=0
    for i_thousand_episode in range(n_thousand_episode):
        goal_reached_times=0
        check_extrinsic_reward=0
        for i_episode in range(check_frequence):
            overall_episode +=1
            episode_length = 0
            current_state,rule_state = env.reset()
            totalspeed=0
            current_state=current_state.reshape(-1)
            done = False
            time_i=0
            while not done:
                current_state=current_state.reshape(-1)

                #print(suboptimal_low_action)
                #print(stocatsic(suboptimal_low_action))
                #suboptimal_low_action
                #suoptimal_probs= suoptimal_probs.clone().detach().to(device)        
                goal_reached = False
                total_timestep += 1
                episode_length += 1                   
                joint_state_goal = np.concatenate([current_state,[env.lat_center()],[env.vehicle.on_road]])    
                action,agent_probs = agent.select_action(joint_state_goal)
                #print(action)
                #print(agent_probs)

                next_state, env_reward, done, _ = env.step(action) #high_level_reward,low_level_reward,



                    
                totalspeed+=env.egovehiclespeed()
                stats.episode_rewards[i_thousand_episode*check_frequence + i_episode] += env_reward 
                stats.episode_low_level_reward[i_thousand_episode*check_frequence + i_episode]+=env_reward
                stats.episode_lengths[i_thousand_episode*check_frequence + i_episode] = episode_length
                goal_reached=0
                stats.goal_reached_times[i_thousand_episode*check_frequence + i_episode]+=goal_reached                              
                goal_reached_times+=goal_reached                    
                next_state=next_state.reshape(-1)                    
                joint_next_state_goal = np.concatenate([next_state,[env.lat_center()],[env.vehicle.on_road]])              
                agent.update_controller(offline_replay)
                check_extrinsic_reward+= env_reward
                time_i=time_i+1
                if not done:
                    current_state = next_state
                else:
                    stats.trap[i_thousand_episode*check_frequence + i_episode]=env.out_of_trap()
                    stats.average_speed[i_thousand_episode*check_frequence + i_episode]= totalspeed/episode_length
                    stats.distance[i_thousand_episode*check_frequence + i_episode]= totalspeed
                    stats.crash_times[i_thousand_episode*check_frequence + i_episode]=env.reach_end_crash()
                if done:
                    print(
                        "episode:{} | length:{} | reward:{:.2f} | crash:{}".format(
                            overall_episode,
                            episode_length,
                            stats.episode_rewards[i_thousand_episode*check_frequence + i_episode],
                            env.reach_end_crash(),
                        )
                    )
                # Goal Finished
                #这里添加平均最好值


        #writer.add_scalar("hdqnp1", stats.episode_env_rewards[i_thousand_episode*check_frequence + i_episode], i_thousand_episode*check_frequence + i_episode)
        print("goal_reached_times:{}".format(goal_reached_times))
        intrinsic_thousand_reward_totall.append(goal_reached_times)
        print("mean_extrinsic_reward:{}".format(check_extrinsic_reward/check_frequence))
        extrinsic_thousand_reward_totall.append(check_extrinsic_reward/check_frequence)    
    
        
        if (check_extrinsic_reward/check_frequence)>=max(extrinsic_thousand_reward_totall):
            best_reward=check_extrinsic_reward/check_frequence
            agent.save_model()
            best_episode=i_thousand_episode

        print("save best model in {}, best mean reward{}".format(best_episode,best_reward))

    return agent, stats, intrinsic_thousand_reward_totall# , visits
    
