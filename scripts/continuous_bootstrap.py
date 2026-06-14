import argparse
import contextlib
import io
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

with contextlib.redirect_stdout(io.StringIO()):
    import highway_env  # noqa: F401

from Dscontroller import DSLaneChangeController
from highway_env import utils


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "continuous_results"


@dataclass
class ExperimentConfig:
    policy_frequency: int = 5
    simulation_frequency: int = 15
    duration: int = 150
    acceleration_range: Tuple[float, float] = (-1.0, 1.0)
    steering_range: Tuple[float, float] = (-math.pi / 50.0, math.pi / 50.0)
    cruise_speed: float = 15.0
    gamma: float = 0.95
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 128
    replay_size: int = 100000
    offline_fraction: float = 0.6
    initial_alpha: float = 0.05
    target_entropy: float = -2.0
    kl_weight: float = 0.5
    reward_kl_penalty: float = 0.05
    ds_std: Tuple[float, float] = (0.10, 0.10)
    training_expert_blend: float = 0.0
    training_noise: float = 0.01
    offline_episodes: int = 80
    bc_steps: int = 0
    train_episodes: int = 3000
    log_interval: int = 150
    seed: int = 41
    model_path: str = str(OUTPUT_DIR / "continuous_low_level_controller.pth")
    stats_path: str = str(OUTPUT_DIR / "continuous_hdqn_train_stats.csv")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(cfg: ExperimentConfig, seed: int = None):
    try:
        env = gym.make("highway-fast-v0", disable_env_checker=True)
    except TypeError:
        env = gym.make("highway-fast-v0")
    env.configure(
        {
            "simulation_frequency": cfg.simulation_frequency,
            "policy_frequency": cfg.policy_frequency,
            "duration": cfg.duration,
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": cfg.acceleration_range,
                "steering_range": cfg.steering_range,
                "longitudinal": True,
                "lateral": True,
                "clip": True,
            },
        }
    )
    if seed is not None:
        env.seed(seed)
    env.reset()
    return env


def flatten_obs(obs: np.ndarray, env) -> np.ndarray:
    return np.concatenate([obs.reshape(-1), [env.lat_center()], [env.vehicle.on_road]]).astype(np.float32)


def normalize_action(value: float, bounds: Tuple[float, float]) -> float:
    return float(np.clip(utils.lmap(value, bounds, [-1.0, 1.0]), -1.0, 1.0))


def atanh_clip(action, eps: float = 1e-5):
    action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(action) - torch.log1p(-action))


def ds_action_penalty(action: np.ndarray, ds_action: np.ndarray, cfg: ExperimentConfig) -> float:
    action_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
    ds_action_t = torch.as_tensor(ds_action, dtype=torch.float32).unsqueeze(0)
    ds_std = torch.as_tensor(cfg.ds_std, dtype=torch.float32).unsqueeze(0)
    z = atanh_clip(action_t)
    ds_z = atanh_clip(ds_action_t)
    return float((0.5 * ((z - ds_z) / ds_std).pow(2)).sum(dim=1).item())


def expert_target(model: DSLaneChangeController, env, rule_state: np.ndarray, cfg: ExperimentConfig) -> Tuple[float, int]:
    with contextlib.redirect_stdout(io.StringIO()):
        target_speed, target_lane, _rl_weight = model.control(True, env, rule_state.reshape(-1))
    if target_speed >= 12.5:
        target_speed = cfg.cruise_speed
    target_lane = int(np.clip(target_lane, 0, env.config["lanes_count"] - 1))
    return float(target_speed), target_lane


def expert_action(env, target_speed: float, target_lane: int, cfg: ExperimentConfig) -> np.ndarray:
    vehicle = env.vehicle
    lane_id = int(np.clip(target_lane, 0, env.config["lanes_count"] - 1))
    lane_index = (vehicle.lane_index[0], vehicle.lane_index[1], lane_id)
    target = env.road.network.get_lane(lane_index)
    lane_coords = target.local_coordinates(vehicle.position)
    lane_next_coords = lane_coords[0] + vehicle.speed
    lane_future_heading = target.heading_at(lane_next_coords)

    lateral_speed_command = -0.30 * lane_coords[1]
    heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(vehicle.speed), -1.0, 1.0))
    heading_ref = lane_future_heading + np.clip(heading_command, -np.pi, np.pi)
    heading_rate_command = 2.0 * utils.wrap_to_pi(heading_ref - vehicle.heading)
    slip_angle = np.arcsin(
        np.clip(vehicle.LENGTH / 2.0 / utils.not_zero(vehicle.speed) * heading_rate_command, -1.0, 1.0)
    )
    steering = np.arctan(2.0 * np.tan(slip_angle))
    acceleration = 0.60 * (target_speed - vehicle.speed)

    acceleration = np.clip(acceleration, *cfg.acceleration_range)
    steering = np.clip(steering, *cfg.steering_range)
    return np.array(
        [
            normalize_action(acceleration, cfg.acceleration_range),
            normalize_action(steering, cfg.steering_range),
        ],
        dtype=np.float32,
    )


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        transition = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            float(done),
        )
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            np.stack(states),
            np.stack(actions),
            np.stack(next_states),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.memory)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        hidden = self.net(state)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -5.0, 1.0)
        return mean, log_std

    def sample(self, state, deterministic: bool = False):
        mean, log_std = self.forward(state)
        if deterministic:
            raw = mean
        else:
            raw = Normal(mean, log_std.exp()).rsample()
        action = torch.tanh(raw)
        if deterministic:
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
        else:
            normal = Normal(mean, log_std.exp())
            log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _log_std = self.forward(state)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class ContinuousSAC:
    def __init__(self, state_dim: int, action_dim: int, cfg: ExperimentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = GaussianActor(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.target_q1 = QNetwork(state_dim, action_dim).to(device)
        self.target_q2 = QNetwork(state_dim, action_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(cfg.initial_alpha), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.state_mean = None
        self.state_std = None

    def set_state_normalizer(self, replay: ReplayBuffer) -> None:
        states = np.stack([transition[0] for transition in replay.memory]).astype(np.float32)
        mean = states.mean(axis=0)
        std = states.std(axis=0)
        std = np.where(std < 1e-3, 1.0, std)
        self.state_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.state_std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

    def normalize_states(self, states):
        if self.state_mean is None or self.state_std is None:
            return states
        return (states - self.state_mean) / self.state_std

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        tensor = self.normalize_states(tensor)
        with torch.no_grad():
            action, _ = self.actor.sample(tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def behavior_clone(self, replay: ReplayBuffer, steps: int, batch_size: int) -> None:
        if len(replay) < batch_size:
            return
        for _ in range(steps):
            states, actions, _next_states, _rewards, _dones = replay.sample(batch_size)
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            states_t = self.normalize_states(states_t)
            actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            loss = self.ds_kl_loss(states_t, actions_t)
            self.actor_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()

    def ds_kl_loss(self, normalized_states, ds_actions) -> torch.Tensor:
        mean_pi, log_std_pi = self.actor(normalized_states)
        std_pi = log_std_pi.exp()
        mean_ds = atanh_clip(ds_actions)
        std_ds = torch.as_tensor(self.cfg.ds_std, dtype=torch.float32, device=self.device).view(1, -1)
        log_std_ds = torch.log(std_ds)
        kl = (
            log_std_pi
            - log_std_ds
            + (std_ds.pow(2) + (mean_ds - mean_pi).pow(2)) / (2.0 * std_pi.pow(2))
            - 0.5
        )
        return kl.sum(dim=1).mean()

    def update(self, offline: ReplayBuffer, online: ReplayBuffer, bc_replay: ReplayBuffer = None) -> Dict[str, float]:
        batch_size = self.cfg.batch_size
        off_n = min(int(batch_size * self.cfg.offline_fraction), len(offline))
        on_n = batch_size - off_n
        if len(offline) < off_n or len(online) < on_n or off_n == 0:
            return {}

        off = offline.sample(off_n)
        if on_n:
            on = online.sample(on_n)
            states = np.concatenate([off[0], on[0]], axis=0)
            actions = np.concatenate([off[1], on[1]], axis=0)
            next_states = np.concatenate([off[2], on[2]], axis=0)
            rewards = np.concatenate([off[3], on[3]], axis=0)
            dones = np.concatenate([off[4], on[4]], axis=0)
        else:
            states, actions, next_states, rewards, dones = off

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        states_t = self.normalize_states(states_t)
        next_states_t = self.normalize_states(next_states_t)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states_t)
            target_q = torch.min(self.target_q1(next_states_t, next_action), self.target_q2(next_states_t, next_action))
            target = rewards_t + self.cfg.gamma * (1.0 - dones_t) * (target_q - self.alpha.detach() * next_log_prob)

        q1_loss = F.mse_loss(self.q1(states_t, actions_t), target)
        q2_loss = F.mse_loss(self.q2(states_t, actions_t), target)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob = self.actor.sample(states_t)
        min_q = torch.min(self.q1(states_t, new_action), self.q2(states_t, new_action))
        actor_loss = (self.alpha.detach() * log_prob - min_q).mean()
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.kl_weight > 0:
            source = bc_replay if bc_replay is not None and len(bc_replay) >= off_n else offline
            bc_states, bc_actions, _bc_next, _bc_rewards, _bc_dones = source.sample(off_n)
            bc_states = torch.as_tensor(bc_states, dtype=torch.float32, device=self.device)
            bc_states = self.normalize_states(bc_states)
            bc_actions = torch.as_tensor(bc_actions, dtype=torch.float32, device=self.device)
            kl_loss = self.ds_kl_loss(bc_states, bc_actions)
            actor_loss = actor_loss + self.cfg.kl_weight * kl_loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.cfg.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update(self.q1, self.target_q1)
        self.soft_update(self.q2, self.target_q2)
        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    def soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.cfg.tau) + param.data * self.cfg.tau)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "state_mean": self.state_mean.detach().cpu() if self.state_mean is not None else None,
                "state_std": self.state_std.detach().cpu() if self.state_std is not None else None,
                "config": asdict(self.cfg),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        if "q1" in checkpoint:
            self.q1.load_state_dict(checkpoint["q1"])
        if "q2" in checkpoint:
            self.q2.load_state_dict(checkpoint["q2"])
        if checkpoint.get("state_mean") is not None:
            self.state_mean = checkpoint["state_mean"].to(self.device)
            self.state_std = checkpoint["state_std"].to(self.device)


def run_episode(
    env,
    agent: ContinuousSAC,
    ds_model: DSLaneChangeController,
    cfg: ExperimentConfig,
    train: bool,
    offline: ReplayBuffer = None,
    online: ReplayBuffer = None,
    bc_replay: ReplayBuffer = None,
) -> Dict[str, float]:
    obs, rule_state = env.reset()
    state = flatten_obs(obs, env)
    total_reward = 0.0
    total_speed = 0.0
    length = 0
    done = False
    last_losses = {}

    while not done:
        target_speed, target_lane = expert_target(ds_model, env, rule_state, cfg)
        target_action = expert_action(env, target_speed, target_lane, cfg)
        action = agent.select_action(state, deterministic=not train)
        if train:
            action = np.clip(
                cfg.training_expert_blend * target_action
                + (1.0 - cfg.training_expert_blend) * action
                + np.random.normal(0.0, cfg.training_noise, size=action.shape),
                -1.0,
                1.0,
            ).astype(np.float32)
        next_obs, reward, done, info = env.step(action)
        next_state = flatten_obs(next_obs, env)
        kl_penalty = ds_action_penalty(action, target_action, cfg)
        shaped_reward = reward - cfg.reward_kl_penalty * kl_penalty
        if online is not None:
            online.push(state, action, next_state, shaped_reward, done)
        if train and bc_replay is not None:
            bc_replay.push(state, target_action, next_state, reward, done)
        if train and offline is not None and online is not None:
            last_losses = agent.update(offline, online, bc_replay) or last_losses
        total_reward += reward
        total_speed += env.egovehiclespeed()
        length += 1
        state = next_state
        rule_state = info["rule_state"]

    return {
        "reward": total_reward,
        "length": float(length),
        "average_speed": total_speed / max(length, 1),
        "overtake": float(env.out_of_trap()),
        "crash": float(env.reach_end_crash()),
        **last_losses,
    }


def collect_offline(cfg: ExperimentConfig):
    env = make_env(cfg, cfg.seed)
    ds_model = DSLaneChangeController(ds_threshold=60.0)
    replay = ReplayBuffer(cfg.replay_size)
    bc_replay = ReplayBuffer(cfg.replay_size)
    for episode in range(cfg.offline_episodes):
        env.seed(cfg.seed + 1000 + episode)
        obs, rule_state = env.reset()
        state = flatten_obs(obs, env)
        done = False
        while not done:
            target_speed, target_lane = expert_target(ds_model, env, rule_state, cfg)
            action = expert_action(env, target_speed, target_lane, cfg)
            next_obs, reward, done, info = env.step(action)
            next_state = flatten_obs(next_obs, env)
            replay.push(state, action, next_state, reward, done)
            bc_replay.push(state, action, next_state, reward, done)
            state = next_state
            rule_state = info["rule_state"]
        if (episode + 1) % 10 == 0 or episode + 1 == cfg.offline_episodes:
            print(f"offline_episode={episode + 1} replay_size={len(replay)}", flush=True)
    env.close()
    return replay, bc_replay


def train(cfg: ExperimentConfig, device: torch.device) -> ContinuousSAC:
    set_seed(cfg.seed)
    offline, bc_replay = collect_offline(cfg)
    env = make_env(cfg, cfg.seed + 2000)
    state_dim = int(np.prod(env.observation_space.shape)) + 2
    action_dim = int(env.action_space.shape[0])
    agent = ContinuousSAC(state_dim, action_dim, cfg, device)
    agent.set_state_normalizer(offline)
    if cfg.bc_steps > 0:
        agent.behavior_clone(offline, cfg.bc_steps, cfg.batch_size)
    else:
        print("bc_warmstart=disabled", flush=True)
    agent.save(cfg.model_path)
    online = ReplayBuffer(cfg.replay_size)
    ds_model = DSLaneChangeController(ds_threshold=60.0)
    stats = []
    best_score = 0.0
    best_metrics = {"source": "online_sac_no_bc_warmstart"}

    for episode in range(cfg.train_episodes):
        env.seed(cfg.seed + 3000 + episode)
        metrics = run_episode(env, agent, ds_model, cfg, train=True, offline=offline, online=online, bc_replay=bc_replay)
        stats.append(metrics)
        recent = stats[-20:]
        score = (
            np.mean([m["overtake"] for m in recent])
            + 0.03 * np.mean([m["average_speed"] for m in recent])
            - 2.0 * np.mean([m["crash"] for m in recent])
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            agent.save(cfg.model_path)
        if (episode + 1) % cfg.log_interval == 0 or episode + 1 == cfg.train_episodes:
            recent_arr = summarize_metrics(recent)
            print(
                f"episode={episode + 1} "
                f"overtake={recent_arr['overtake_rate']:.3f} "
                f"crash={recent_arr['crash_rate']:.3f} "
                f"speed={recent_arr['average_speed']:.3f} "
                f"reward={recent_arr['reward']:.3f} "
                f"alpha={metrics.get('alpha', float('nan')):.4f}",
                flush=True,
            )

    env.close()
    save_table(cfg.stats_path, stats)
    agent.load(cfg.model_path)
    print(f"saved_best_model={cfg.model_path} best_episode_metrics={best_metrics}", flush=True)
    return agent


def summarize_metrics(stats):
    return {
        "episodes": len(stats),
        "reward": float(np.mean([m["reward"] for m in stats])),
        "average_speed": float(np.mean([m["average_speed"] for m in stats])),
        "overtake_rate": float(np.mean([m["overtake"] for m in stats])),
        "crash_rate": float(np.mean([m["crash"] for m in stats])),
        "length": float(np.mean([m["length"] for m in stats])),
    }


def save_table(path: str, stats) -> None:
    keys = ["reward", "length", "average_speed", "overtake", "crash", "q1_loss", "q2_loss", "actor_loss", "kl_loss", "alpha"]
    rows = []
    for metric in stats:
        rows.append([metric.get(key, np.nan) for key in keys])
    np.savetxt(path, np.asarray(rows, dtype=np.float64), delimiter=",", header=",".join(keys), comments="")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--offline-episodes", type=int, default=None)
    parser.add_argument("--bc-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--reward-kl-penalty", type=float, default=None)
    parser.add_argument("--ds-std", type=float, nargs=2, default=None)
    parser.add_argument("--training-expert-blend", type=float, default=None)
    parser.add_argument("--training-noise", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    if args.train_episodes is not None:
        cfg.train_episodes = args.train_episodes
    if args.offline_episodes is not None:
        cfg.offline_episodes = args.offline_episodes
    if args.bc_steps is not None:
        cfg.bc_steps = args.bc_steps
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.seed is not None:
        cfg.seed = args.seed
    if args.kl_weight is not None:
        cfg.kl_weight = args.kl_weight
    if args.reward_kl_penalty is not None:
        cfg.reward_kl_penalty = args.reward_kl_penalty
    if args.ds_std is not None:
        cfg.ds_std = tuple(args.ds_std)
    if args.training_expert_blend is not None:
        cfg.training_expert_blend = args.training_expert_blend
    if args.training_noise is not None:
        cfg.training_noise = args.training_noise
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.duration is not None:
        cfg.duration = args.duration
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.stats_path is not None:
        cfg.stats_path = args.stats_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train(cfg, device)


if __name__ == "__main__":
    main()
