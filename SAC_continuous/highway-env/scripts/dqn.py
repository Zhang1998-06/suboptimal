import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import gym
from utils.replay_memory import ReplayMemory
from utils import plotting
from gym.envs.registration import register
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle





def dqn_learning(
    env,
    agent,
    num_episodes,
    ):

    """The h-DQN learning algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    agent:
        a h-DQN agent consists of a meta-controller and controller.
    num_episodes:
        Number (can be divided by 1000) of episodes to run for. Ex: 12000
    exploration_schedule: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    gamma: float
        Discount Factor
    """
    ###############
    # RUN ENV     #
    ###############
    # Keep track of useful statistics
    check_frequence=5
     
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        crash_times=np.zeros(num_episodes),
        trap=np.zeros(num_episodes),
        average_speed=np.zeros(num_episodes),
        distance=np.zeros(num_episodes),
        )
    n_thousand_episode = int(np.floor(num_episodes / check_frequence))
    '''
    visits = np.zeros((n_thousand_episode, agent.observation_in_features))#是做存贮用吗,state 有可能是one_hot这样差别很大。episode,state6   now spisode,observation 25
    '''
    total_timestep = 0
    best_reward=0
    overall_episode=0
    thousand_episode_reward=[]
    best_episode=0
    for i_thousand_episode in range(n_thousand_episode):

        check_frequence_reward =0

        for i_episode in range(check_frequence):
            overall_episode +=1
            episode_length = 0
            current_state = env.reset()
            totalspeed=0
            done = False
            outoftrap=False
            while not done:
                # Get annealing exploration rate (epislon) from exploration_schedule
                total_timestep += 1
                episode_length += 1
                current_state=current_state.reshape(-1)
                new_current_state=np.concatenate([current_state,[env.lat_center()],[env.vehicle.on_road]])    
                    # Get annealing exploration rate (epislon) from exploration_schedule                
                action = agent.select_action(new_current_state)    
                ### Step the env and store the transition
                next_state, reward, done, _ = env.step(action)
                totalspeed+=env.egovehiclespeed()
                # Update statistics
                stats.episode_rewards[i_thousand_episode*check_frequence + i_episode] += reward
                stats.episode_lengths[i_thousand_episode*check_frequence + i_episode] = episode_length
                
                check_frequence_reward+=reward
                #visits[i_thousand_episode][next_state-1] += 1
                next_state=next_state.reshape(-1)      
                new_next_state=np.concatenate([next_state,[env.lat_center()],[env.vehicle.on_road]])  
                agent.replay_memory.push(new_current_state, action, new_next_state, reward, done)
                # Update Both meta-controller and controller
                agent.update_controller()               
                current_state = next_state
                if done:
                    stats.trap[i_thousand_episode*check_frequence + i_episode]=env.out_of_trap()
                    stats.average_speed[i_thousand_episode*check_frequence + i_episode]= totalspeed/episode_length
                    stats.distance[i_thousand_episode*check_frequence + i_episode]= totalspeed
                    stats.crash_times[i_thousand_episode*check_frequence + i_episode]=env.reach_end_crash()

            print("episode:{}".format(i_thousand_episode*check_frequence + i_episode))  
            print("extrinsic_reward:{}".format(stats.episode_rewards[i_thousand_episode*check_frequence + i_episode]))  
        
        thousand_episode_reward.append(check_frequence_reward)
        if check_frequence_reward>=max(thousand_episode_reward):
            best_reward=check_frequence_reward/check_frequence
            agent.save_model()
            best_episode=i_thousand_episode

        print("save best model in {}, best mean reward{}".format(best_episode,best_reward))
                
    return agent, stats# , visits
    