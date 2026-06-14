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
    def __init__(self, in_features, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #print("other dim")
        #print(self.fc2(x))
        return self.fc2(x)

class ValueNet(nn.Module):
    def __init__(self, in_features):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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
        control_path='low_level_controller.pth'
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

        # Actor (policy) network
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)

        # Critics (Q networks) and targets
        self.critic_1 = QValueNet(self.state_dim, self.action_dim).to(device)
        self.critic_2 = QValueNet(self.state_dim, self.action_dim).to(device)
        self.value_net = ValueNet(self.state_dim).to(device)
        self.target_value_net = ValueNet(self.state_dim).to(device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # In IQL we need a value network trained via expectile regression.
        # We reuse the alpha_lr so major hyper-parameters stay intact.
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=alpha_lr)

        # Hyperparameters
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        # IQL-specific scalars
        self.expectile = 0.7
        self.policy_temperature = 3.0
        self.max_weight = 100.0

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
            next_v = self.target_value_net(next_states)
            td_target = rewards + self.gamma * next_v * (1 - dones)
        return td_target

    def expectile_loss(self, value, target):
        diff = target - value
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def update_controller(self, offline_replay):
        if len(offline_replay) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = offline_replay.sample(self.batch_size)

        # Convert to torch
        states_var = Variable(torch.from_numpy(states).type(dtype)).to(self.device)
        actions_var = Variable(torch.from_numpy(actions).long()).to(self.device).unsqueeze(1)
        next_var = Variable(torch.from_numpy(next_states).type(dtype)).to(self.device)
        rewards_var = Variable(torch.from_numpy(rewards).type(dtype)).to(self.device).unsqueeze(1)
        dones_var = Variable(torch.from_numpy(dones)).type(dtype).to(self.device).unsqueeze(1)

        # 1) Value update via expectile regression
        critic_1_q = self.critic_1(states_var).gather(1, actions_var)
        critic_2_q = self.critic_2(states_var).gather(1, actions_var)
        min_q = torch.min(critic_1_q, critic_2_q).detach()
        value = self.value_net(states_var)
        value_loss = self.expectile_loss(value, min_q)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 2) Critic update using bootstrapped value
        td_target = self.calc_target(rewards_var, next_var, dones_var)
        critic_1_loss = F.mse_loss(critic_1_q, td_target.detach())
        critic_2_loss = F.mse_loss(critic_2_q, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 3) Policy update via advantage-weighted regression
        probs = self.actor(states_var)
        log_probs = torch.log(probs + 1e-8)
        log_prob_actions = log_probs.gather(1, actions_var)
        with torch.no_grad():
            v = self.value_net(states_var)
            adv = torch.min(critic_1_q.detach(), critic_2_q.detach()) - v
            weights = torch.exp(adv / self.policy_temperature)
            weights = torch.clamp(weights, max=self.max_weight)
        actor_loss = -(weights * log_prob_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4) Soft update target value
        self.soft_update(self.value_net, self.target_value_net)

    def save_model(self, path: Optional[str] = None):
        save_path = path or self.control_path
        torch.save({'actor': self.actor.state_dict()}, save_path)

    def load(self, path: Optional[str] = None):
        load_path = path or self.control_path
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])



