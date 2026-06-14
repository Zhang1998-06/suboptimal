import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from continuous_bootstrap import (
    ContinuousSAC,
    ExperimentConfig,
    ReplayBuffer,
    flatten_obs,
    make_env,
    save_table,
    set_seed,
    summarize_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "continuous_results"
PREFIX = OUTPUT_DIR / "continuous_vanilla_sac_3000"


def validate_gamma(gamma: float) -> float:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    return gamma


def update_online(agent: ContinuousSAC, replay: ReplayBuffer) -> Dict[str, float]:
    batch_size = agent.cfg.batch_size
    if len(replay) < batch_size:
        return {}

    states, actions, next_states, rewards, dones = replay.sample(batch_size)
    states_t = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=agent.device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=agent.device).unsqueeze(1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=agent.device).unsqueeze(1)

    states_t = agent.normalize_states(states_t)
    next_states_t = agent.normalize_states(next_states_t)

    with torch.no_grad():
        next_action, next_log_prob = agent.actor.sample(next_states_t)
        target_q = torch.min(
            agent.target_q1(next_states_t, next_action),
            agent.target_q2(next_states_t, next_action),
        )
        target = rewards_t + agent.cfg.gamma * (1.0 - dones_t) * (
            target_q - agent.alpha.detach() * next_log_prob
        )

    q1_loss = F.mse_loss(agent.q1(states_t, actions_t), target)
    q2_loss = F.mse_loss(agent.q2(states_t, actions_t), target)

    agent.q1_opt.zero_grad()
    q1_loss.backward()
    agent.q1_opt.step()

    agent.q2_opt.zero_grad()
    q2_loss.backward()
    agent.q2_opt.step()

    new_action, log_prob = agent.actor.sample(states_t)
    min_q = torch.min(agent.q1(states_t, new_action), agent.q2(states_t, new_action))
    actor_loss = (agent.alpha.detach() * log_prob - min_q).mean()

    agent.actor_opt.zero_grad()
    actor_loss.backward()
    agent.actor_opt.step()

    alpha_loss = -(agent.log_alpha * (log_prob + agent.cfg.target_entropy).detach()).mean()
    agent.alpha_opt.zero_grad()
    alpha_loss.backward()
    agent.alpha_opt.step()

    agent.soft_update(agent.q1, agent.target_q1)
    agent.soft_update(agent.q2, agent.target_q2)

    return {
        "q1_loss": float(q1_loss.detach().cpu()),
        "q2_loss": float(q2_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "kl_loss": 0.0,
        "alpha": float(agent.alpha.detach().cpu()),
    }


def sample_random_action(env) -> np.ndarray:
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def run_episode(
    env,
    agent: ContinuousSAC,
    cfg: ExperimentConfig,
    train: bool,
    replay: Optional[ReplayBuffer] = None,
    random_steps_remaining: int = 0,
) -> Dict[str, float]:
    obs, _rule_state = env.reset()
    state = flatten_obs(obs, env)
    total_reward = 0.0
    total_speed = 0.0
    length = 0
    done = False
    last_losses: Dict[str, float] = {}
    random_actions = 0

    while not done:
        if train and random_steps_remaining > 0:
            action = sample_random_action(env)
            random_steps_remaining -= 1
            random_actions += 1
        else:
            action = agent.select_action(state, deterministic=not train)

        next_obs, reward, done, _info = env.step(action)
        next_state = flatten_obs(next_obs, env)

        if replay is not None:
            replay.push(state, action, next_state, reward, done)
        if train and replay is not None:
            last_losses = update_online(agent, replay) or last_losses

        total_reward += float(reward)
        total_speed += float(env.egovehiclespeed())
        length += 1
        state = next_state

    return {
        "reward": total_reward,
        "length": float(length),
        "average_speed": total_speed / max(length, 1),
        "overtake": float(env.out_of_trap()),
        "crash": float(env.reach_end_crash()),
        "random_actions": float(random_actions),
        **last_losses,
    }


def train(
    cfg: ExperimentConfig,
    device: torch.device,
    random_steps: int,
    log_interval: int,
    checkpoint_selection: str,
) -> ContinuousSAC:
    set_seed(cfg.seed)
    env = make_env(cfg, cfg.seed + 2000)
    state_dim = int(np.prod(env.observation_space.shape)) + 2
    action_dim = int(env.action_space.shape[0])
    agent = ContinuousSAC(state_dim, action_dim, cfg, device)
    replay = ReplayBuffer(cfg.replay_size)
    stats = []
    best_score = -math.inf
    best_metrics: Dict[str, float] = {"source": "vanilla_online_sac"}
    remaining_random_steps = random_steps

    if checkpoint_selection == "best":
        agent.save(cfg.model_path)

    for episode in range(cfg.train_episodes):
        env.seed(cfg.seed + 3000 + episode)
        metrics = run_episode(
            env,
            agent,
            cfg,
            train=True,
            replay=replay,
            random_steps_remaining=remaining_random_steps,
        )
        remaining_random_steps = max(0, remaining_random_steps - int(metrics["random_actions"]))
        stats.append(metrics)

        recent = stats[-20:]
        score = (
            np.mean([m["overtake"] for m in recent])
            + 0.03 * np.mean([m["average_speed"] for m in recent])
            - 2.0 * np.mean([m["crash"] for m in recent])
        )
        if score > best_score:
            best_score = float(score)
            best_metrics = metrics
            if checkpoint_selection == "best":
                agent.save(cfg.model_path)

        if (episode + 1) % log_interval == 0 or episode + 1 == cfg.train_episodes:
            save_table(cfg.stats_path, stats)
            recent_arr = summarize_metrics(recent)
            print(
                f"episode={episode + 1} "
                f"overtake={recent_arr['overtake_rate']:.3f} "
                f"crash={recent_arr['crash_rate']:.3f} "
                f"speed={recent_arr['average_speed']:.3f} "
                f"reward={recent_arr['reward']:.3f} "
                f"alpha={metrics.get('alpha', float('nan')):.4f} "
                f"replay_size={len(replay)}",
                flush=True,
            )

    env.close()
    save_table(cfg.stats_path, stats)
    if checkpoint_selection == "best":
        agent.load(cfg.model_path)
        print(
            f"saved_best_model={cfg.model_path} best_score={best_score:.6f} "
            f"best_episode_metrics={best_metrics}",
            flush=True,
        )
    else:
        agent.save(cfg.model_path)
        print(
            f"saved_final_model={cfg.model_path} best_score_observed={best_score:.6f} "
            f"best_episode_metrics_observed={best_metrics}",
            flush=True,
        )
    return agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--policy-frequency", type=int, default=5)
    parser.add_argument("--simulation-frequency", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--model-path", type=str, default=str(PREFIX) + "_controller.pth")
    parser.add_argument("--stats-path", type=str, default=str(PREFIX) + "_train_stats.csv")
    parser.add_argument("--checkpoint-selection", choices=["best", "final"], default="best")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.train_episodes = args.train_episodes
    cfg.seed = args.seed
    cfg.gamma = validate_gamma(args.gamma)
    cfg.duration = args.duration
    cfg.policy_frequency = args.policy_frequency
    cfg.simulation_frequency = args.simulation_frequency
    cfg.batch_size = args.batch_size
    cfg.replay_size = args.replay_size
    cfg.offline_episodes = 0
    cfg.offline_fraction = 0.0
    cfg.bc_steps = 0
    cfg.kl_weight = 0.0
    cfg.reward_kl_penalty = 0.0
    cfg.training_expert_blend = 0.0
    cfg.training_noise = 0.0
    cfg.model_path = args.model_path
    cfg.stats_path = args.stats_path

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for output_path in (cfg.model_path, cfg.stats_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(cfg.seed)
    train(cfg, device, args.random_steps, args.log_interval, args.checkpoint_selection)


if __name__ == "__main__":
    main()
