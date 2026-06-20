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
    def __init__(self, in_features, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor([1.0, np.pi / 50], dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias


class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        control_path: Optional[str] = None,
    ):
        self.env = env
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.control_path = control_path or 'low_level_controller.pth'
        self.device = device

        self.discrete_action_map = np.array([
            (-1.0, -np.pi / 50),
            (-1.0, 0.0),
            (-1.0, np.pi / 50),
            (0.0, -np.pi / 50),
            (0.0, 0.0),
            (0.0, np.pi / 50),
            (1.0, -np.pi / 50),
            (1.0, 0.0),
            (1.0, np.pi / 50),
        ], dtype=np.float32)
        self.discrete_action_tensor = torch.tensor(self.discrete_action_map, dtype=torch.float32, device=self.device)
        self.max_steer = np.pi / 50
        self.action_low_bound = torch.tensor([-1.0, -self.max_steer], dtype=torch.float32, device=self.device)
        self.action_high_bound = torch.tensor([1.0, self.max_steer], dtype=torch.float32, device=self.device)

        # We flatten the obs space
        obs_example = np.zeros(env.observation_space.shape)
        self.observation_in_features = obs_example.size
        # We'll add 2 extra dims: [env.lat_center(), env.vehicle.on_road] from your code
        self.state_dim = self.observation_in_features + 2
        self.action_dim = 2

        # Actor (policy) network
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = PolicyNet(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critics (Q networks) and targets
        self.critic_1 = QValueNet(self.state_dim, self.action_dim).to(self.device)
        self.critic_2 = QValueNet(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_1 = QValueNet(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_2 = QValueNet(self.state_dim, self.action_dim).to(self.device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Hyperparameters
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.bc_alpha = 2.5
        self.total_it = 0

    def select_action(self, joint_state_goal):
        state_tensor = torch.from_numpy(joint_state_goal).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        discrete_action = self._continuous_to_discrete(action)
        return discrete_action, action

    def _continuous_to_discrete(self, action):
        diffs = self.discrete_action_map - action
        idx = np.argmin(np.linalg.norm(diffs, axis=1))
        return int(idx)

    def _indices_to_action_vectors(self, indices_tensor):
        return self.discrete_action_tensor[indices_tensor]

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update_controller(self, offline_replay):
        if len(offline_replay) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = offline_replay.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).float().unsqueeze(1).to(self.device)
        actions = np.array(actions)
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        action_indices = actions.astype(np.int64)
        action_indices_tensor = torch.from_numpy(action_indices).long().to(self.device)
        action_vectors = self._indices_to_action_vectors(action_indices_tensor)

        self.total_it += 1

        with torch.no_grad():
            next_action = self.actor_target(next_states)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = torch.max(torch.min(next_action + noise, self.action_high_bound), self.action_low_bound)
            target_q1 = self.target_critic_1(next_states, next_action)
            target_q2 = self.target_critic_2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic_1(states, action_vectors)
        current_q2 = self.critic_2(states, action_vectors)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi_actions = self.actor(states)
            q_pi = self.critic_1(states, pi_actions)
            bc_loss = F.mse_loss(pi_actions, action_vectors)
            lambda_coef = self.bc_alpha / (q_pi.abs().mean().detach() + 1e-6)
            actor_loss = -lambda_coef * q_pi.mean() + bc_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)
            self.soft_update(self.actor, self.actor_target)

    def save_model(self):
        torch.save({'actor': self.actor.state_dict()}, self.control_path)

    def load(self):
        checkpoint = torch.load(self.control_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(self.actor.state_dict())



