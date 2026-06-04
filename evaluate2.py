"""
Step 11 revised: Evaluation and visualization script with parameters configured in code.

Save as:
    evaluate.py

Required project files:
    assignment/target_assignment.py
    envs/uav_env.py
    algorithms/actor_critic.py
    algorithms/rollout_buffer.py
    algorithms/mappo.py

Usage in PyCharm:
    1. Open this file.
    2. Modify the CONFIG section below.
    3. Right click evaluate.py -> Run 'evaluate'.

No command-line parameters are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from step6 import MultiUAV2DEnv, UAVEnvConfig
from mappo import MAPPOAgent, MAPPOConfig


# =============================================================================
# CONFIG: modify these parameters directly in PyCharm
# =============================================================================
@dataclass
class EvalConfig:
    # Checkpoint path.
    # Change this to the model you want to evaluate.
    checkpoint: str = "runs/stage2_10_obstacles/checkpoints/best.pt"

    # Output.
    output_dir: str = "eval_outputs"
    save_plots: bool = True
    show_plots: bool = False

    # Evaluation.
    episodes: int = 10
    deterministic: bool = True
    device: str = "auto"  # "auto", "cpu", or "cuda"
    seed: int = 123

    # Environment. Keep these consistent with training.
    num_agents: int = 3
    num_targets: int = 3
    num_obstacles: int = 10
    max_steps: int = 600
    assigner_name: str = "hungarian"  # "hungarian", "greedy", or "fixed"
    lidar_num_rays: int = 35
    lidar_range: float = 5.0

    # Network config. Keep these consistent with training.
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    activation: str = "tanh"


CFG = EvalConfig()


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(cfg: EvalConfig, seed_offset: int = 0) -> MultiUAV2DEnv:
    env_cfg = UAVEnvConfig(
        num_agents=cfg.num_agents,
        num_targets=cfg.num_targets,
        num_obstacles=cfg.num_obstacles,
        max_steps=cfg.max_steps,
        assigner_name=cfg.assigner_name,
        lidar_num_rays=cfg.lidar_num_rays,
        lidar_range=cfg.lidar_range,
        seed=cfg.seed + seed_offset,
    )
    return MultiUAV2DEnv(env_cfg)


def make_agent_for_loading(env: MultiUAV2DEnv, cfg: EvalConfig) -> MAPPOAgent:
    """
    Build an agent with matching environment dimensions, then load checkpoint.

    The optimizer hyperparameters do not matter for pure evaluation, but MAPPOAgent
    needs a config to construct the model.
    """
    mappo_cfg = MAPPOConfig(
        rollout_steps=32,
        ppo_epochs=1,
        minibatch_size=32,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        activation=cfg.activation,
        device=cfg.device,
    )
    agent = MAPPOAgent(env, mappo_cfg)

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train first or update CFG.checkpoint in evaluate.py."
        )

    agent.load(checkpoint_path)
    agent.model.eval()
    return agent


def run_episode(
    agent: MAPPOAgent,
    env: MultiUAV2DEnv,
    seed: int | None,
    deterministic: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray | str]]:
    """Run one evaluation episode and collect trajectory data."""
    obs = env.reset(seed=seed)
    done = False

    positions_history: List[np.ndarray] = [env.positions.copy()]
    assignments_history: List[np.ndarray] = [env.assignments.copy()]
    arrived_history: List[np.ndarray] = [env.arrived.copy()]
    reward_history: List[np.ndarray] = []

    episode_return = np.zeros(env.num_agents, dtype=np.float32)
    episode_length = 0
    info: Dict = {}

    while not done:
        actions = agent.act(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(actions)

        episode_return += rewards
        episode_length += 1
        reward_history.append(rewards.copy())
        positions_history.append(env.positions.copy())
        assignments_history.append(env.assignments.copy())
        arrived_history.append(env.arrived.copy())

        done = bool(np.all(dones))

    reason = info.get("termination_reason", "")
    success = float("success_all_arrived" in reason)
    collision_or_boundary = float("collision" in reason or "boundary_violation" in reason)
    timeout = float("timeout" in reason)

    metrics = {
        "episode_return_mean": float(np.mean(episode_return)),
        "episode_return_sum": float(np.sum(episode_return)),
        "episode_length": float(episode_length),
        "success": success,
        "collision_or_boundary": collision_or_boundary,
        "timeout": timeout,
    }

    trajectory_data = {
        "positions_history": np.asarray(positions_history, dtype=np.float32),
        "assignments_history": np.asarray(assignments_history, dtype=np.int64),
        "arrived_history": np.asarray(arrived_history, dtype=bool),
        "reward_history": np.asarray(reward_history, dtype=np.float32),
        "target_positions": env.target_positions.copy(),
        "obstacle_centers": env.obstacle_centers.copy(),
        "obstacle_radii": env.obstacle_radii.copy(),
        "final_assignments": env.assignments.copy(),
        "termination_reason": reason,
    }
    return metrics, trajectory_data


def plot_trajectory(
    trajectory_data: Dict[str, np.ndarray | str],
    env_cfg: UAVEnvConfig,
    output_path: Path | None = None,
    show: bool = False,
    title: str = "DA-MAPPO Evaluation Trajectory",
) -> None:
    """Plot UAV trajectories, targets, obstacles, and final assignments."""
    import matplotlib.pyplot as plt

    positions_history = trajectory_data["positions_history"]
    arrived_history = trajectory_data["arrived_history"]
    target_positions = trajectory_data["target_positions"]
    obstacle_centers = trajectory_data["obstacle_centers"]
    obstacle_radii = trajectory_data["obstacle_radii"]
    final_assignments = trajectory_data["final_assignments"]
    reason = trajectory_data["termination_reason"]

    num_agents = positions_history.shape[1]
    half_size = env_cfg.world_size / 2.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\ntermination: {reason}")
    ax.grid(True)

    # Obstacles.
    for center, radius in zip(obstacle_centers, obstacle_radii):
        circle = plt.Circle(center, radius, fill=True, alpha=0.45)
        ax.add_patch(circle)

    # Targets.
    ax.scatter(target_positions[:, 0], target_positions[:, 1], marker="*", s=180, label="Targets")
    for t_id, target in enumerate(target_positions):
        ax.text(target[0], target[1], f"T{t_id}")

    # UAV trajectories.
    # If a UAV reaches its target, only draw its trajectory up to the first arrival step.
    # This avoids plotting the post-arrival drift/hover part, making the figure cleaner.
    for i in range(num_agents):
        full_traj = positions_history[:, i, :]

        arrived_indices = np.where(arrived_history[:, i])[0]
        if len(arrived_indices) > 0:
            end_idx = int(arrived_indices[0])
        else:
            end_idx = full_traj.shape[0] - 1

        traj = full_traj[: end_idx + 1]
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f"UAV {i}")
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=80)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=100)

        assigned_target = target_positions[final_assignments[i]]
        ax.plot(
            [traj[-1, 0], assigned_target[0]],
            [traj[-1, 1], assigned_target[1]],
            linestyle="--",
            linewidth=1,
        )
        ax.text(traj[-1, 0], traj[-1, 1], f"U{i}")

    ax.legend(loc="upper left")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate episode metrics."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float32)
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated


# =============================================================================
# Main evaluation
# =============================================================================
def main() -> None:
    cfg = CFG
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, seed_offset=0)
    agent = make_agent_for_loading(env, cfg)

    print("DA-MAPPO evaluation started")
    print(f"checkpoint: {cfg.checkpoint}")
    print(f"device: {agent.device}")
    print(f"episodes: {cfg.episodes}")
    print(f"num_agents: {env.num_agents}")
    print(f"num_targets: {env.num_targets}")
    print(f"num_obstacles: {cfg.num_obstacles}")
    print(f"obs_dim: {agent.obs_dim}")
    print(f"state_dim: {agent.state_dim}")
    print("-" * 80)

    all_metrics: List[Dict[str, float]] = []

    for ep in range(cfg.episodes):
        episode_seed = cfg.seed + ep
        metrics, trajectory_data = run_episode(
            agent=agent,
            env=env,
            seed=episode_seed,
            deterministic=cfg.deterministic,
        )
        all_metrics.append(metrics)

        reason = trajectory_data["termination_reason"]
        print(
            f"episode={ep + 1:03d} | "
            f"return_mean={metrics['episode_return_mean']:.3f} | "
            f"length={metrics['episode_length']:.0f} | "
            f"success={metrics['success']:.0f} | "
            f"reason={reason}"
        )

        if cfg.save_plots or cfg.show_plots:
            plot_path = output_dir / f"trajectory_ep_{ep + 1:03d}.png" if cfg.save_plots else None
            plot_trajectory(
                trajectory_data=trajectory_data,
                env_cfg=env.cfg,
                output_path=plot_path,
                show=cfg.show_plots,
                title=f"DA-MAPPO Evaluation Episode {ep + 1}",
            )

    aggregated = aggregate_metrics(all_metrics)
    print("-" * 80)
    print("Aggregated metrics:")
    for key, value in aggregated.items():
        print(f"{key}: {value:.6f}")

    if cfg.save_plots:
        print(f"Trajectory plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
