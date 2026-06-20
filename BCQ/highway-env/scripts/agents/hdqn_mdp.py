from re import X
import numpy as np
import random
from collections import namedtuple
import os
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
    """Simple behavior cloning network used by BCQ to imitate the dataset policy."""

    def __init__(self, in_features, out_features):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = torch.clamp(logits, min=-10, max=10)
        if logits.dim() == 1:
            return F.softmax(logits, dim=0)
        return F.softmax(logits, dim=1)

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
        batch_size=128,
        control_path='low_level_controller.pth',
    ):
        self.env = env
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.control_path = control_path
        self.action_dim = env.action_space.n

        # We flatten the obs space
        obs_example = np.zeros(env.observation_space.shape)
        self.observation_in_features = obs_example.size
        # We'll add 2 extra dims: [env.lat_center(), env.vehicle.on_road] from your code
        self.state_dim = self.observation_in_features + 2

        # Behavior cloning network used to filter admissible actions
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
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # BCQ-specific parameters
        self.bcq_threshold = 0.3

    def select_action(self, joint_state_goal):
        joint_state_goal = torch.from_numpy(joint_state_goal).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(joint_state_goal)
            q1_value = self.critic_1(joint_state_goal)
            q2_value = self.critic_2(joint_state_goal)
            min_qvalue = torch.min(q1_value, q2_value)
            action_idx, _ = self._bcq_action_selection(min_qvalue, probs)
        return action_idx.item(), probs.squeeze(0)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            next_probs = self.actor(next_states)
            q1_value = self.target_critic_1(next_states)
            q2_value = self.target_critic_2(next_states)
            min_qvalue = torch.min(q1_value, q2_value)
            _, filtered_q = self._bcq_action_selection(min_qvalue, next_probs)
            td_target = rewards + self.gamma * filtered_q * (1 - dones)
        return td_target

    def _bcq_action_selection(self, q_values, behavior_probs):
        """Filter actions using behavior cloning probabilities and select best Q."""
        mask = behavior_probs >= self.bcq_threshold
        large_neg = torch.full_like(q_values, -1e9)
        filtered_q = torch.where(mask, q_values, large_neg)
        max_values, max_indices = torch.max(filtered_q, dim=1, keepdim=True)
        fallback_actions = torch.argmax(behavior_probs, dim=1, keepdim=True)
        valid = mask.any(dim=1, keepdim=True)
        chosen_actions = torch.where(valid, max_indices, fallback_actions)
        chosen_values = q_values.gather(1, chosen_actions)
        return chosen_actions.squeeze(1), chosen_values

    def update_controller(self, offline_replay):
        if len(offline_replay) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = offline_replay.sample(self.batch_size)

        states_var = Variable(torch.from_numpy(states).float())
        actions_var = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        next_var = Variable(torch.from_numpy(next_states).float())
        rewards_var = Variable(torch.from_numpy(rewards.astype(np.float32))).unsqueeze(1)
        dones_var = Variable(torch.from_numpy(dones.astype(np.float32))).unsqueeze(1)

        td_target = self.calc_target(rewards_var, next_var, dones_var)

        critic_1_q = self.critic_1(states_var).gather(1, actions_var)
        critic_2_q = self.critic_2(states_var).gather(1, actions_var)

        critic_1_loss = F.mse_loss(critic_1_q, td_target.detach())
        critic_2_loss = F.mse_loss(critic_2_q, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        behavior_probs = self.actor(states_var)
        selected_action_probs = behavior_probs.gather(1, actions_var)
        behavior_loss = -(torch.log(selected_action_probs + 1e-8)).mean()

        self.actor_optimizer.zero_grad()
        behavior_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self):
        os.makedirs(os.path.dirname(self.control_path) or '.', exist_ok=True)
        torch.save(
            {
                'actor': self.actor.state_dict(),
                'critic_1': self.critic_1.state_dict(),
                'critic_2': self.critic_2.state_dict(),
            },
            self.control_path,
        )

    def load(self):
        checkpoint = torch.load(self.control_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        if 'critic_1' in checkpoint:
            self.critic_1.load_state_dict(checkpoint['critic_1'])
        if 'critic_2' in checkpoint:
            self.critic_2.load_state_dict(checkpoint['critic_2'])
        # Ensure all networks reside on the execution device after loading
        self.actor.to(self.device)
        self.critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())



