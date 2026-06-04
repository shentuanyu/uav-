"""
Step 10: Training entry point for DA-MAPPO reproduction.

Save as:
    train.py

Required project files:
    assignment/target_assignment.py
    envs/uav_env.py
    algorithms/actor_critic.py
    algorithms/rollout_buffer.py
    algorithms/mappo.py

This script provides:
- Environment construction
- MAPPOAgent construction
- Full training loop
- Periodic console logging
- Periodic evaluation rollout
- Periodic checkpoint saving
- CUDA/GPU automatic support through MAPPOAgent(device="auto")

Recommended first test:
    python train.py --total-updates 20 --rollout-steps 128 --num-obstacles 5

Longer run:
    python train.py --total-updates 1000 --rollout-steps 256 --num-obstacles 10
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from step7 import MultiUAV2DEnv, UAVEnvConfig
from mappo import MAPPOAgent, MAPPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DA-MAPPO on the multi-UAV target assignment environment.")

    # General.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", type=str, default="stage3.2_15_obstacles")
    parser.add_argument("--save-dir", type=str, default="runs3")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="runs3/stage3.1_15_obstacles/checkpoints/final.pt",
        help="Path to checkpoint for continuing training."
    )

    # Environment.
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--assigner-name", type=str, default="hungarian", choices=["hungarian", "greedy", "fixed"])
    parser.add_argument("--lidar-num-rays", type=int, default=35)
    parser.add_argument("--lidar-range", type=float, default=5.0)

    # Training.
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Network.
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "gelu", "elu"])

    # Logging / saving / evaluation.
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--deterministic-eval", action="store_true", default=True)



    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(args: argparse.Namespace, seed_offset: int = 0) -> MultiUAV2DEnv:
    cfg = UAVEnvConfig(
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        num_obstacles=args.num_obstacles,
        max_steps=args.max_steps,
        assigner_name=args.assigner_name,
        lidar_num_rays=args.lidar_num_rays,
        lidar_range=args.lidar_range,
        seed=args.seed + seed_offset,
    )
    return MultiUAV2DEnv(cfg)


def make_agent(env: MultiUAV2DEnv, args: argparse.Namespace) -> MAPPOAgent:
    cfg = MAPPOConfig(
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        activation=args.activation,
        device=args.device,
    )
    return MAPPOAgent(env, cfg)


def evaluate_policy(
    agent: MAPPOAgent,
    env: MultiUAV2DEnv,
    num_episodes: int = 5,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Run evaluation episodes without training."""
    episode_returns = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    for ep in range(num_episodes):
        obs = env.reset(seed=env.cfg.seed + 10_000 + ep if env.cfg.seed is not None else None)
        done = False
        ep_return = np.zeros(env.num_agents, dtype=np.float32)
        ep_len = 0
        info = {}

        while not done:
            actions = agent.act(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(actions)
            ep_return += rewards
            ep_len += 1
            done = bool(np.all(dones))

        reason = info.get("termination_reason", "")
        if "success_all_arrived" in reason:
            success_count += 1
        if "collision" in reason or "boundary_violation" in reason:
            collision_count += 1
        if "timeout" in reason:
            timeout_count += 1

        episode_returns.append(float(np.mean(ep_return)))
        episode_lengths.append(float(ep_len))

    return {
        "eval_mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "eval_std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "eval_mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "eval_success_rate": float(success_count / max(num_episodes, 1)),
        "eval_collision_rate": float(collision_count / max(num_episodes, 1)),
        "eval_timeout_rate": float(timeout_count / max(num_episodes, 1)),
    }


def write_csv_row(csv_path: Path, row: Dict[str, float]) -> None:
    """Append one metrics row to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format selected metrics for console output."""
    keys = [
        "update",
        "total_env_steps",
        "mean_reward",
        "mean_episode_return",
        "episodes_finished",
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "eval_mean_return",
        "eval_success_rate",
    ]
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if key == "update":
                parts.append(f"{key}={int(value)}")
            else:
                parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.save_dir) / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    csv_path = run_dir / "metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args, seed_offset=0)
    eval_env = make_env(args, seed_offset=999)
    agent = make_agent(env, args)
    if args.resume_checkpoint:
        print(f"Loading checkpoint from: {args.resume_checkpoint}")
        agent.load(args.resume_checkpoint)
        print(f"Resume training from update={agent.num_updates}, env_steps={agent.total_env_steps}")

    print("DA-MAPPO training started")
    print(f"run_dir: {run_dir}")
    print(f"device: {agent.device}")
    print(f"num_agents: {agent.num_agents}")
    print(f"obs_dim: {agent.obs_dim}")
    print(f"state_dim: {agent.state_dim}")
    print(f"action_dim: {agent.action_dim}")
    print(f"rollout_steps: {args.rollout_steps}")
    print(f"num_obstacles: {args.num_obstacles}")
    print("-" * 80)

    start_time = time.time()
    best_eval_return = -float("inf")

    for update in range(1, args.total_updates + 1):
        train_metrics = agent.train_one_update()

        metrics: Dict[str, float] = {
            "update": float(update),
            "time_sec": float(time.time() - start_time),
            **train_metrics,
        }

        should_eval = args.eval_interval > 0 and update % args.eval_interval == 0
        if should_eval:
            eval_metrics = evaluate_policy(
                agent=agent,
                env=eval_env,
                num_episodes=args.eval_episodes,
                deterministic=args.deterministic_eval,
            )
            metrics.update(eval_metrics)

            if eval_metrics["eval_mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["eval_mean_return"]
                agent.save(checkpoint_dir / "best.pt")
                metrics["saved_best"] = 1.0
            else:
                metrics["saved_best"] = 0.0

        should_save = args.save_interval > 0 and update % args.save_interval == 0
        if should_save:
            agent.save(checkpoint_dir / f"update_{update}.pt")

        write_csv_row(csv_path, metrics)

        if args.log_interval > 0 and update % args.log_interval == 0:
            print(format_metrics(metrics))

    agent.save(checkpoint_dir / "final.pt")
    print("-" * 80)
    print("Training finished")
    print(f"Final checkpoint: {checkpoint_dir / 'final.pt'}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
