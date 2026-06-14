from re import X
import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import gym
from utils.replay_memory import ReplayMemory, Transition
import functools
import itertools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable, List
from gym import spaces
import numpy as np

from highway_env import utils
from highway_env.utils import Vector
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.envs.common.action import GOAL
if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv
USE_CUDA = torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class PolicyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x=self.fc2(x)
        #print("here is the shape")
        #print(x.shape)
        #print("Logits:", x)
        x = torch.clamp(x, min=-10, max=10)
        if x.dim() == 1:
            #print(F.softmax(x, dim=0))
            return F.softmax(x, dim=0)
        else:
            return F.softmax(x, dim=1)

class QValueNet(nn.Module):
    def __init__(self, in_features,action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #print("other dim")
        #print(self.fc2(x))
        return self.fc2(x)

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
class HSAC():

    def __init__(self, env, actor_lr, critic_lr,alpha_lr, target_entropy, tau, gamma, device,replay_memory_size=10000, batch_size=128):
        ###############
        # BUILD MODEL #
        ###############
        self.env = env
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.control_path = 'low_level_controller.pth'
        self.action_dim = env.action_space.n  # action space
        self.GOAL_out_features = 9
        
        self.observation_in_feature = np.zeros(env.observation_space.shape)
        self.observation_in_features = sum(len(row) for row in self.observation_in_feature)   # observation space
        self.state_dim= self.observation_in_features+ 2
        # 策略网络
        self.actor = PolicyNet(self.state_dim,  self.action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(self.state_dim, self.action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(self.state_dim, self.action_dim).to(device)
        self.target_critic_1 = QValueNet(self.state_dim, 
                                         self.action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(self.state_dim, 
                                         self.action_dim).to(device)  # 第二个目标Q网络

        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Construct the optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # Construct the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
         # this beta is difference between Q value
        self.log_beta = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_beta.requires_grad = True  # 可以对alpha求梯度
        self.log_beta_optimizer = torch.optim.Adam([self.log_beta],
                                                    lr=alpha_lr)       
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device




    def select_action(self, joint_state_goal):
        joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
        with torch.no_grad():
            probs = self.actor(Variable(joint_state_goal))
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return action.cpu().item(), probs




    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target        
    

    def update_controller(self, transition_dict=None):
        if transition_dict:
            # Optionally push the data from GAIL into HSAC’s own replay buffer
            for s, a, r, ns, d in zip(
                transition_dict["states"],
                transition_dict["actions"],
                transition_dict["rewards"],
                transition_dict["next_states"],
                transition_dict["dones"]
            ):
                self.replay_memory.push(s, a, ns, r, d)        
        if len(self.replay_memory) < self.batch_size:
            return

        state_goal_batch, action_batch, next_state_goal_batch, reward_batch, done_mask = \
            self.replay_memory.sample(self.batch_size)
        
        states = Variable(torch.from_numpy(state_goal_batch).type(dtype)).to(self.device)
 
        action_batch = Variable(torch.from_numpy(action_batch).long()).to(self.device).unsqueeze(1) 
     
        next_states = Variable(torch.from_numpy(next_state_goal_batch).type(dtype)).to(self.device)
 
        rewards= Variable(torch.from_numpy(reward_batch).type(dtype)).to(self.device).unsqueeze(1) 
  
        dones = Variable(torch.from_numpy(done_mask)).type(dtype).to(self.device).unsqueeze(1) 
 

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
 
        critic_1_q_values = self.critic_1(states).gather(1, action_batch)
 
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, action_batch)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self):
        torch.save({
            'actor': self.actor.state_dict(),
        }, self.control_path)


    def load(self):
        checkpoint = torch.load(self.control_path)
        self.actor.load_state_dict(checkpoint['actor'])



