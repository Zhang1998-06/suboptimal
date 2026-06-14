# bc_agent.py

from utils import plotting  # Ensure you have a utils/plotting.py with EpisodeStats
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    """
    Example policy network for discrete actions.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Convert logits to probabilities for discrete actions
        return F.softmax(x, dim=-1)


class BehaviorClone:
    """
    Behavior Cloning agent for discrete actions [0..action_dim-1].
    Uses cross-entropy to match expert actions.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, lr=1e-3):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        """
        Supervised learning update:
          states: shape (batch_size, state_dim)
          actions: shape (batch_size,) of integers in [0..action_dim-1]
        """
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)

        # Forward pass: produce raw logits
        logits = self.policy(states_t)
        # Cross-entropy will internally do log_softmax + NLL
        bc_loss = F.cross_entropy(logits, actions_t)

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        """
        Chooses an action by sampling from the categorical distribution
        given by softmax(logits).
        state: shape (state_dim,)
        Returns: int in [0..action_dim-1]
        """
        # Convert to 2D tensor: (1, state_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_t)                # shape (1, action_dim)
            probs = F.softmax(logits, dim=-1)            # shape (1, action_dim)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()                    # shape (1,)
        return action.item()


@dataclass
class SafetyMetricTracker:
    """Track headway, TTC, lateral jerk, and lane invasions during an episode."""

    dt: float
    min_headway: float = np.inf
    min_ttc: float = np.inf
    max_lateral_jerk: float = 0.0
    lane_invasion_time: float = 0.0
    prev_lat: Optional[float] = None
    prev_lat_vel: Optional[float] = None
    prev_lat_acc: Optional[float] = None

    def update(self, env) -> None:
        vehicle = getattr(env, "vehicle", None)
        if vehicle is None:
            return

        front_vehicle, _ = env.road.neighbour_vehicles(vehicle)
        if front_vehicle is not None:
            headway = vehicle.lane_distance_to(front_vehicle)
            if headway > 0:
                self.min_headway = min(self.min_headway, headway)
                ego_speed = vehicle.speed * np.cos(vehicle.heading)
                front_speed = front_vehicle.speed * np.cos(front_vehicle.heading)
                relative_speed = ego_speed - front_speed
                if relative_speed > 1e-3:
                    ttc = headway / relative_speed
                    self.min_ttc = min(self.min_ttc, ttc)

        lane = vehicle.lane
        if lane is None:
            return
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        lane_width = lane.width_at(longitudinal)
        if abs(lateral) > lane_width / 2.0:
            self.lane_invasion_time += self.dt

        if self.prev_lat is not None:
            lat_vel = (lateral - self.prev_lat) / self.dt
            if self.prev_lat_vel is not None:
                lat_acc = (lat_vel - self.prev_lat_vel) / self.dt
                if self.prev_lat_acc is not None:
                    lat_jerk = (lat_acc - self.prev_lat_acc) / self.dt
                    self.max_lateral_jerk = max(self.max_lateral_jerk, abs(lat_jerk))
                self.prev_lat_acc = lat_acc
            self.prev_lat_vel = lat_vel
        self.prev_lat = lateral


def bc_learning(
    env,
    bc_agent,
    expert_s,
    expert_a,
    num_episodes,
    batch_size=64,
    check_frequence=5
):
    """
    Offline BC training + environment rollouts for evaluation.
    We'll store stats in EpisodeStats from your `utils.plotting`.
    """
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        crash_times=np.zeros(num_episodes),
        trap=np.zeros(num_episodes),
        average_speed=np.zeros(num_episodes),
        distance=np.zeros(num_episodes),
        episode_low_level_reward=np.zeros(num_episodes),
        goal_reached_times=np.zeros(num_episodes),
        episode_min_headway=np.full(num_episodes, np.nan),
        episode_min_ttc=np.full(num_episodes, np.nan),
        episode_max_lateral_jerk=np.zeros(num_episodes),
        episode_lane_invasion=np.zeros(num_episodes),
    )

    n_group = num_episodes // check_frequence
    overall_episode = 0
    best_mean_return = -1e9
    best_episode = 0

    for group_idx in range(n_group):
        group_reward_sum = 0.0

        for _ in range(check_frequence):
            overall_episode += 1

            # (A) BC update from offline data
            sample_indices = np.random.randint(low=0, high=expert_s.shape[0], size=batch_size)
            batch_s = expert_s[sample_indices]
            batch_a = expert_a[sample_indices]
            bc_agent.learn(batch_s, batch_a)

            # (B) Evaluate policy in env
            # Flatten or adapt the state if needed
            current_state,rule_state = env.reset()
            done = False
            ep_length = 0
            ep_reward = 0.0
            total_speed = 0.0
            config = getattr(env, "config", {}) or {}
            policy_frequency = config.get("policy_frequency") or config.get("simulation_frequency", 1)
            dt = 1.0 / policy_frequency if policy_frequency else 1.0
            safety_tracker = SafetyMetricTracker(dt=dt)

            while not done:

                current_state=current_state.reshape(-1)
                joint_state_goal = np.concatenate([current_state,[env.lat_center()],[env.vehicle.on_road]]) 


                action = bc_agent.take_action(joint_state_goal)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                # Only call this if your env actually implements egovehiclespeed()
                # Otherwise, comment it out or define your speed measure
                total_speed += env.egovehiclespeed()
                safety_tracker.update(env)

                # Move on
                next_state = next_state.reshape(-1)
                current_state = next_state
            idx = overall_episode - 1
            stats.episode_rewards[idx] = ep_reward
            stats.episode_lengths[idx] = ep_length
            # Watch out for dividing by zero
            stats.average_speed[idx] = total_speed / max(ep_length, 1e-8)
            stats.distance[idx] = total_speed

            # If your env has these
            stats.trap[idx] = env.out_of_trap()
            stats.crash_times[idx] = env.reach_end_crash()
            stats.episode_low_level_reward[idx] = ep_reward
            stats.goal_reached_times[idx] = info.get("goal_reached", 0)
            stats.episode_min_headway[idx] = (
                safety_tracker.min_headway if np.isfinite(safety_tracker.min_headway) else np.nan
            )
            stats.episode_min_ttc[idx] = (
                safety_tracker.min_ttc if np.isfinite(safety_tracker.min_ttc) else np.nan
            )
            stats.episode_max_lateral_jerk[idx] = safety_tracker.max_lateral_jerk
            stats.episode_lane_invasion[idx] = safety_tracker.lane_invasion_time

            group_reward_sum += ep_reward

        mean_return = group_reward_sum / check_frequence
        print(f"[BC] Group {group_idx+1}/{n_group} | Episodes {overall_episode-check_frequence+1}~{overall_episode} | "
              f"Mean Reward: {mean_return:.3f}")

        if mean_return > best_mean_return:
            best_mean_return = mean_return
            best_episode = overall_episode
            # Optionally save the BC policy
            # torch.save(bc_agent.policy.state_dict(), "best_bc_model.pth")
            model_save_path = f"best_bc_model.pth"
            torch.save(bc_agent.policy.state_dict(), model_save_path)
            print(f" [*] Best model updated at episode {best_episode} with mean reward {best_mean_return:.3f}")

    return bc_agent, stats
