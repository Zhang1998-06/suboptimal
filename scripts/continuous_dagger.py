import argparse
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from continuous_bootstrap import (
    ExperimentConfig,
    DSLaneChangeController,
    expert_action,
    expert_target,
    flatten_obs,
    make_env,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "continuous_results"


@dataclass
class DaggerConfig:
    offline_episodes: int = 50
    initial_bc_steps: int = 6000
    dagger_iterations: int = 6
    dagger_episodes: int = 15
    dagger_bc_steps: int = 2500
    batch_size: int = 256
    replay_size: int = 200000
    lr: float = 3e-4
    seed: int = 11
    training_noise: float = 0.02
    model_path: str = str(OUTPUT_DIR / "continuous_dagger_controller.pth")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ImitationReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.position = 0

    def push(self, state, action):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
        else:
            self.states[self.position] = state
            self.actions[self.position] = action
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.states), size=batch_size)
        states = np.stack([self.states[i] for i in idx])
        actions = np.stack([self.actions[i] for i in idx])
        return states, actions

    def arrays(self):
        return np.stack(self.states), np.stack(self.actions)

    def __len__(self) -> int:
        return len(self.states)


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def set_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=self.state_mean.device))
        self.state_std.copy_(torch.as_tensor(std, dtype=torch.float32, device=self.state_std.device))

    def forward(self, state):
        state = (state - self.state_mean) / self.state_std
        return self.net(state)


class DaggerAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: DaggerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.policy = DeterministicPolicy(state_dim, action_dim).to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def update_normalizer(self, replay: ImitationReplay) -> None:
        states, _actions = replay.arrays()
        mean = states.mean(axis=0)
        std = states.std(axis=0)
        std = np.where(std < 1e-3, 1.0, std)
        self.policy.set_normalizer(mean, std)

    def train_bc(self, replay: ImitationReplay, steps: int) -> float:
        self.update_normalizer(replay)
        last_loss = 0.0
        for _ in range(steps):
            states, actions = replay.sample(self.cfg.batch_size)
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            loss = F.mse_loss(self.policy(states_t), actions_t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy(state_t).squeeze(0).cpu().numpy().astype(np.float32)

    def save(self, path: str, exp_cfg: ExperimentConfig) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "dagger_config": asdict(self.cfg),
                "experiment_config": asdict(exp_cfg),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])


def rollout(env, agent, exp_cfg: ExperimentConfig, replay: ImitationReplay = None, blend: float = 0.0, noise: float = 0.0):
    ds_model = DSLaneChangeController(ds_threshold=60.0)
    obs, rule_state = env.reset()
    state = flatten_obs(obs, env)
    done = False
    total_reward = 0.0
    total_speed = 0.0
    length = 0
    while not done:
        target_speed, target_lane = expert_target(ds_model, env, rule_state, exp_cfg)
        target_action = expert_action(env, target_speed, target_lane, exp_cfg)
        if replay is not None:
            replay.push(state, target_action)
        if agent is None:
            action = target_action
        else:
            policy_action = agent.select_action(state)
            action = blend * target_action + (1.0 - blend) * policy_action
            if noise > 0:
                action = action + np.random.normal(0.0, noise, size=action.shape)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_speed += env.egovehiclespeed()
        length += 1
        state = flatten_obs(next_obs, env)
        rule_state = info["rule_state"]
    return {
        "reward": total_reward,
        "length": float(length),
        "average_speed": total_speed / max(length, 1),
        "overtake": float(env.out_of_trap()),
        "crash": float(env.reach_end_crash()),
    }


def collect_expert(exp_cfg: ExperimentConfig, cfg: DaggerConfig, replay: ImitationReplay) -> None:
    env = make_env(exp_cfg, cfg.seed)
    for episode in range(cfg.offline_episodes):
        env.seed(cfg.seed + 1000 + episode)
        rollout(env, None, exp_cfg, replay=replay)
        if (episode + 1) % 10 == 0 or episode + 1 == cfg.offline_episodes:
            print(f"expert_episode={episode + 1} replay_size={len(replay)}", flush=True)
    env.close()


def train(exp_cfg: ExperimentConfig, cfg: DaggerConfig, device: torch.device) -> DaggerAgent:
    set_seed(cfg.seed)
    replay = ImitationReplay(cfg.replay_size)
    collect_expert(exp_cfg, cfg, replay)
    env = make_env(exp_cfg, cfg.seed + 2000)
    state_dim = int(np.prod(env.observation_space.shape)) + 2
    action_dim = int(env.action_space.shape[0])
    agent = DaggerAgent(state_dim, action_dim, cfg, device)
    loss = agent.train_bc(replay, cfg.initial_bc_steps)
    print(f"initial_bc_loss={loss:.6f}", flush=True)

    for iteration in range(cfg.dagger_iterations):
        blend = max(0.0, 0.8 * (1.0 - iteration / max(cfg.dagger_iterations - 1, 1)))
        metrics = []
        for episode in range(cfg.dagger_episodes):
            env.seed(cfg.seed + 3000 + iteration * cfg.dagger_episodes + episode)
            metrics.append(rollout(env, agent, exp_cfg, replay=replay, blend=blend, noise=cfg.training_noise))
        loss = agent.train_bc(replay, cfg.dagger_bc_steps)
        summary = summarize(metrics)
        print(
            f"dagger_iter={iteration + 1} blend={blend:.2f} "
            f"overtake={summary['overtake_rate']:.3f} crash={summary['crash_rate']:.3f} "
            f"speed={summary['average_speed']:.3f} loss={loss:.6f} replay_size={len(replay)}",
            flush=True,
        )
    env.close()
    agent.save(cfg.model_path, exp_cfg)
    return agent


def summarize(stats):
    return {
        "episodes": len(stats),
        "reward": float(np.mean([m["reward"] for m in stats])),
        "average_speed": float(np.mean([m["average_speed"] for m in stats])),
        "overtake_rate": float(np.mean([m["overtake"] for m in stats])),
        "crash_rate": float(np.mean([m["crash"] for m in stats])),
        "length": float(np.mean([m["length"] for m in stats])),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-episodes", type=int)
    parser.add_argument("--initial-bc-steps", type=int)
    parser.add_argument("--dagger-iterations", type=int)
    parser.add_argument("--dagger-episodes", type=int)
    parser.add_argument("--dagger-bc-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exp_cfg = ExperimentConfig()
    cfg = DaggerConfig()
    for arg_name, field_name in [
        ("offline_episodes", "offline_episodes"),
        ("initial_bc_steps", "initial_bc_steps"),
        ("dagger_iterations", "dagger_iterations"),
        ("dagger_episodes", "dagger_episodes"),
        ("dagger_bc_steps", "dagger_bc_steps"),
        ("seed", "seed"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, field_name, value)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train(exp_cfg, cfg, device)


if __name__ == "__main__":
    main()
