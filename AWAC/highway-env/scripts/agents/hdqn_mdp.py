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
class HCQL():
    def __init__(
        self,
        env,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
        replay_memory_size=10000,
        batch_size=128
    ):
        self.env = env
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.control_path = 'low_level_controller.pth'
        self.action_dim = env.action_space.n

        # We flatten the obs space
        obs_example = np.zeros(env.observation_space.shape)
        self.observation_in_features = obs_example.size
        # We'll add 2 extra dims: [env.lat_center(), env.vehicle.on_road] from your code
        self.state_dim = self.observation_in_features + 2

        # Actor (policy) network
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)

        # Critics (Q networks) and targets
        self.critic_1 = QValueNet(self.state_dim, self.action_dim).to(device)
        self.critic_2 = QValueNet(self.state_dim, self.action_dim).to(device)
        self.target_critic_1 = QValueNet(self.state_dim, self.action_dim).to(device)
        self.target_critic_2 = QValueNet(self.state_dim, self.action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Temperature parameter for SAC part
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Hyperparameters
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # --- AWAC hyperparameters ---
        self.awac_lambda = 1.0  # temperature used in the advantage exponent
        self.awac_max_weight = 20.0  # clamp to avoid exploding weights

    def select_action(self, joint_state_goal):
        joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
        with torch.no_grad():
            probs = self.actor(Variable(joint_state_goal))
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return action.cpu().item(), probs

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            next_probs = self.actor(next_states)
            q1_value = self.target_critic_1(next_states)
            q2_value = self.target_critic_2(next_states)
            min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
            next_value = min_qvalue
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def update_controller(self, offline_replay):
        if len(offline_replay) < self.batch_size:
            return

        # Sample from offline replay only
        states, actions, next_states, rewards, dones = offline_replay.sample(self.batch_size)

        # Convert to torch
        states_var = Variable(torch.from_numpy(states).type(dtype)).to(self.device)
        actions_var = Variable(torch.from_numpy(actions).long()).to(self.device).unsqueeze(1)
        next_var = Variable(torch.from_numpy(next_states).type(dtype)).to(self.device)
        rewards_var = Variable(torch.from_numpy(rewards).type(dtype)).to(self.device).unsqueeze(1)
        dones_var = Variable(torch.from_numpy(dones)).type(dtype).to(self.device).unsqueeze(1)

        td_target = self.calc_target(rewards_var, next_var, dones_var)  # shape: [B, 1]

        # Current Q estimates:
        critic_1_q = self.critic_1(states_var).gather(1, actions_var)  # shape: [B, 1]
        critic_2_q = self.critic_2(states_var).gather(1, actions_var)  # shape: [B, 1]

        critic_1_mse = F.mse_loss(critic_1_q, td_target.detach())
        critic_2_mse = F.mse_loss(critic_2_q, td_target.detach())

        critic_1_loss = critic_1_mse
        critic_2_loss = critic_2_mse

        # Optimize critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Optimize critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ------------------------
        # 2) Actor Update (AWAC)
        # ------------------------
        probs = self.actor(states_var)
        log_probs = torch.log(probs + 1e-8)
        log_prob_actions = log_probs.gather(1, actions_var)

        with torch.no_grad():
            q1_value = self.critic_1(states_var)
            q2_value = self.critic_2(states_var)
            min_qvalue = torch.min(q1_value, q2_value)
            v_value = torch.sum(probs * min_qvalue, dim=1, keepdim=True)
            q_sa = min_qvalue.gather(1, actions_var)
            advantages = q_sa - v_value
            weights = torch.exp(advantages / self.awac_lambda)
            weights = torch.clamp(weights, max=self.awac_max_weight)

        actor_loss = -(weights * log_prob_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------
        # 3) Soft-update target critics
        # ------------------------
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self):
        torch.save({'actor': self.actor.state_dict()}, self.control_path)

    def load(self):
        checkpoint = torch.load(self.control_path)
        self.actor.load_state_dict(checkpoint['actor'])



