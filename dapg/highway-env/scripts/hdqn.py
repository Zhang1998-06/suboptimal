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
    BETA,
    model,
    method="suboptimal"
    ):

    """The h-DQN learning algorithm. load high-level controller, train low-level controller
       the high level controller give the low level controller a prefered goal
       in the set, low level use DQN 
       in other set low level use HDQNstats.episode_low_level_reward
       two set of reward 
    """
    from utils.replay_memory import ReplayMemory
    offline_replay = ReplayMemory(capacity=25000)
    online_replay  = ReplayMemory(capacity=25000)

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
        # Push them into the OFFLINE replay
        offline_replay.push(state, action, next_state, reward, done)
    check_frequence=5
    use_suboptimal_guidance = method.lower() == "suboptimal"
    bc_weight = BETA if method.lower() == "dapg" else 0.0

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        crash_times=np.zeros(num_episodes),
        trap=np.zeros(num_episodes),
        average_speed=np.zeros(num_episodes),
        distance=np.zeros(num_episodes),
        episode_low_level_reward=np.zeros(num_episodes),
        goal_reached_times=np.zeros(num_episodes),
        entropy=np.zeros(num_episodes),
        exp_entropy=np.zeros(num_episodes),
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
            acc_entropy=0
            while not done:
                current_state=current_state.reshape(-1)
                rule_state=rule_state.reshape(-1)
                target_speed,target_lane,RL_weight=model.control(True,env,current_state)
                goal_1=interperate_ds_command(env, target_speed,target_lane)
                goal=get_goal_index(goal_1)
                env.set_goal(goal_1)
                low_level_goal_action=env.goal_low_action()
                low_level_index=get_action_index(env.goal_low_action())
                total_timestep += 1
                episode_length += 1                   
                joint_state_goal = np.concatenate([current_state,[env.lat_center()],[env.vehicle.on_road]])    
                action,agent_probs,actor_entropy = agent.select_action(joint_state_goal)
                acc_entropy+=actor_entropy
                #print(action)
                #print(agent_probs)
                agent_probs= agent_probs.clone().detach().to(device)
                kl_div = torch.tensor(0.0, device=device)
                if use_suboptimal_guidance:
                    suboptimal_low_action= stochastic_multi_actions([low_level_index]).to(device)
                    kl_div = F.kl_div(agent_probs.log(),suboptimal_low_action, reduction='batchmean')
                    kl_div = torch.clamp(kl_div, max=0.5)
                next_state, env_reward, done, _ = env.step(action) #high_level_reward,low_level_reward,
                next_state_rule=_["rule_state"] 
                #kl_div=np.clip(kl_div,0,None)
                RL_weight=np.clip(RL_weight,0.95,1)
                

                newbeta=200*agent.log_alpha.exp().detach()
                new_reward = env_reward- kl_div.item()*newbeta #*RL_weight # the ds treshold is 25m 
                #print("this is the reward")
                #print(new_reward)
                #print(kl_div.item()*BETA)
                # consider collision

                if env_reward>-10:
                    reward_min = 0 - 0.5*newbeta #*RL_weight
                    reward_max = 1
                    new_reward = (new_reward - reward_min) / (reward_max - reward_min)

                if env.vehicle.speed<6:
                    new_reward=0

                    
                totalspeed+=env.egovehiclespeed()
                stats.episode_rewards[i_thousand_episode*check_frequence + i_episode] += env_reward 
                stats.episode_low_level_reward[i_thousand_episode*check_frequence + i_episode]+=env_reward
                stats.episode_lengths[i_thousand_episode*check_frequence + i_episode] = episode_length
                # this goal reached times is the entropy value                                       
                next_state=next_state.reshape(-1)                    
                joint_next_state_goal = np.concatenate([next_state,[env.lat_center()],[env.vehicle.on_road]])              
                online_replay.push(joint_state_goal, action, 
                                   joint_next_state_goal, new_reward, done)
                agent.update_controller(offline_replay, online_replay, bc_weight=bc_weight)
                 
                stats.goal_reached_times[i_thousand_episode*check_frequence + i_episode]+=0
                check_extrinsic_reward+= env_reward
                time_i=time_i+1
                if not done:
                    current_state = next_state
                    rule_state=next_state_rule
                else:
                    stats.trap[i_thousand_episode*check_frequence + i_episode]=env.out_of_trap()
                    stats.crash_times[i_thousand_episode*check_frequence + i_episode]=env.reach_end_crash()
                    stats.average_speed[i_thousand_episode*check_frequence + i_episode]= totalspeed/episode_length
                    stats.distance[i_thousand_episode*check_frequence + i_episode]= totalspeed
                    stats.entropy[i_thousand_episode*check_frequence + i_episode]= acc_entropy/episode_length
                    stats.exp_entropy[i_thousand_episode*check_frequence + i_episode]=  np.exp(acc_entropy/episode_length)
                
                print("done:{}".format(done))
                print("episode:{}".format(overall_episode))
                # Goal Finished
                #这里添加平均最好值

        print("mean_extrinsic_reward:{}".format(check_extrinsic_reward/check_frequence))
        extrinsic_thousand_reward_totall.append(check_extrinsic_reward/check_frequence)    
    
        
        if (check_extrinsic_reward/check_frequence)>=max(extrinsic_thousand_reward_totall):
            best_reward=check_extrinsic_reward/check_frequence
            agent.save_model()
            best_episode=i_thousand_episode

        print("save best model in {}, best mean reward{}".format(best_episode,best_reward))

    return agent, stats, intrinsic_thousand_reward_totall# , visits
    
