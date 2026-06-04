"""
Step 9: MAPPO main algorithm class for DA-MAPPO reproduction.

Save as:
    algorithms/mappo.py

Required files:
    algorithms/actor_critic.py
    algorithms/rollout_buffer.py
    envs/uav_env.py
    assignment/target_assignment.py

This file implements:
- MAPPO hyperparameter config
- rollout collection from the multi-UAV environment
- GAE/return computation through RolloutBuffer
- PPO clipped actor update
- centralized critic update
- entropy regularization
- gradient clipping
- CUDA/GPU support
- model save/load helpers

This is the first complete training algorithm skeleton. It is intentionally kept
compact and readable before adding logging, evaluation, curriculum, and the full
paper reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actor_critic import ActorCritic, NetworkConfig
from rollout_buffer import BufferConfig, RolloutBuffer


@dataclass
class MAPPOConfig:
    """MAPPO hyperparameters."""

    # Rollout / return settings.
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO update settings.
    ppo_epochs: int = 8
    minibatch_size: int = 512
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    use_value_clipping: bool = True

    # Optimizer settings.
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    eps: float = 1e-5

    # Network settings.
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    activation: str = "tanh"
    log_std_init: float = -0.5

    # Device.
    device: str = "auto"


class MAPPOAgent:
    """
    MAPPO trainer for the multi-UAV environment.

    The environment remains numpy/CPU based. Actor-Critic inference and training
    run on CUDA if available.
    """

    def __init__(self, env, config: Optional[MAPPOConfig] = None):
        self.env = env
        self.cfg = config or MAPPOConfig()
        self.device = self._resolve_device(self.cfg.device)

        # Infer dimensions directly from the environment.
        reset_obs = self.env.reset()
        self.num_agents = self.env.num_agents
        self.obs_dim = int(reset_obs.shape[1])
        self.state_dim = int(self.env.get_global_state().shape[0])
        self.action_dim = self.env.action_dim

        network_config = NetworkConfig(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.num_hidden_layers,
            activation=self.cfg.activation,
            log_std_init=self.cfg.log_std_init,
        )
        self.model = ActorCritic(network_config).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(),
            lr=self.cfg.actor_lr,
            eps=self.cfg.eps,
        )
        self.critic_optimizer = optim.Adam(
            self.model.critic.parameters(),
            lr=self.cfg.critic_lr,
            eps=self.cfg.eps,
        )

        buffer_config = BufferConfig(
            rollout_steps=self.cfg.rollout_steps,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            device=str(self.device),
        )
        self.buffer = RolloutBuffer(buffer_config)

        self.total_env_steps = 0
        self.num_updates = 0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_rollout(self) -> Dict[str, float]:
        """
        Collect one rollout into the buffer.

        Returns:
            diagnostics dictionary for this rollout.
        """
        self.buffer.reset()
        obs = self.env.reset()

        episode_returns = []
        episode_lengths = []
        current_episode_return = np.zeros((self.num_agents,), dtype=np.float32)
        current_episode_length = 0

        for _ in range(self.cfg.rollout_steps):
            state = self.env.get_global_state()

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                actions_tensor, log_probs_tensor = self.model.act(obs_tensor, deterministic=False)
                value_tensor = self.model.value(state_tensor)

            actions = actions_tensor.cpu().numpy().astype(np.float32)
            log_probs = log_probs_tensor.cpu().numpy().astype(np.float32)
            value = float(value_tensor.item())

            next_obs, rewards, dones, info = self.env.step(actions)

            self.buffer.add(
                obs=obs,
                state=state,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                value=value,
            )

            self.total_env_steps += self.num_agents
            current_episode_return += rewards
            current_episode_length += 1

            obs = next_obs

            if bool(np.all(dones)):
                episode_returns.append(float(np.mean(current_episode_return)))
                episode_lengths.append(float(current_episode_length))
                obs = self.env.reset()
                current_episode_return[:] = 0.0
                current_episode_length = 0

        # Bootstrap from final state if rollout did not end exactly on terminal.
        final_state = self.env.get_global_state()
        final_state_tensor = torch.as_tensor(final_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(self.model.value(final_state_tensor).item())

        # If the environment ended and was reset inside the loop, the current final state is non-terminal.
        last_done = bool(self.env.done)
        if last_done:
            last_value = 0.0

        self.buffer.compute_returns_and_advantages(last_value=last_value, last_done=last_done)

        diagnostics = self.buffer.summary()
        diagnostics.update(
            {
                "episodes_finished": float(len(episode_returns)),
                "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
                "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "total_env_steps": float(self.total_env_steps),
            }
        )
        return diagnostics

    # ------------------------------------------------------------------
    # PPO / MAPPO update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """Run PPO updates using the current rollout buffer."""
        actor_losses = []
        critic_losses = []
        entropy_values = []
        approx_kls = []
        clip_fractions = []
        total_losses = []

        for _ in range(self.cfg.ppo_epochs):
            for batch in self.buffer.iter_minibatches(
                batch_size=self.cfg.minibatch_size,
                shuffle=True,
                normalize_advantages=self.cfg.normalize_advantages,
            ):
                metrics = self._update_minibatch(batch)
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])
                entropy_values.append(metrics["entropy"])
                approx_kls.append(metrics["approx_kl"])
                clip_fractions.append(metrics["clip_fraction"])
                total_losses.append(metrics["total_loss"])

        self.num_updates += 1
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_values)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "total_loss": float(np.mean(total_losses)),
            "num_updates": float(self.num_updates),
        }

    def _update_minibatch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["old_values"]

        new_log_probs, entropy, new_values = self.model.evaluate_actions(obs, states, actions)

        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        unclipped_policy_loss = -advantages * ratio
        clipped_policy_loss = -advantages * torch.clamp(
            ratio,
            1.0 - self.cfg.clip_coef,
            1.0 + self.cfg.clip_coef,
        )
        actor_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()

        if self.cfg.use_value_clipping:
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.cfg.value_clip_coef,
                self.cfg.value_clip_coef,
            )
            value_losses = (new_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            critic_loss = 0.5 * (new_values - returns).pow(2).mean()

        entropy_loss = entropy.mean()
        total_loss = (
            actor_loss
            + self.cfg.value_loss_coef * critic_loss
            - self.cfg.entropy_coef * entropy_loss
        )

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean()
            )

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy_loss.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_fraction": float(clip_fraction.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def train_one_update(self) -> Dict[str, float]:
        """Collect one rollout and update the policy once."""
        rollout_metrics = self.collect_rollout()
        update_metrics = self.update()
        return {**rollout_metrics, **update_metrics}

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get actions from the current policy for evaluation."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_tensor, _ = self.model.act(obs_tensor, deterministic=deterministic)
        return actions_tensor.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save model and optimizer states."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": self.cfg,
                "total_env_steps": self.total_env_steps,
                "num_updates": self.num_updates,
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "num_agents": self.num_agents,
            },
            path,
        )

    def load(self, path: str | Path, map_location: Optional[str] = None) -> None:
        """Load model and optimizer states."""
        checkpoint = torch.load(path, map_location=map_location or self.device,weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
        self.num_updates = int(checkpoint.get("num_updates", 0))


if __name__ == "__main__":
    # from uav_env import MultiUAV2DEnv, UAVEnvConfig
    from step6 import MultiUAV2DEnv, UAVEnvConfig

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = UAVEnvConfig(
        num_agents=3,
        num_targets=3,
        num_obstacles=5,
        assigner_name="hungarian",
        seed=seed,
    )
    env = MultiUAV2DEnv(env_cfg)

    mappo_cfg = MAPPOConfig(
        rollout_steps=64,
        ppo_epochs=2,
        minibatch_size=64,
        device="auto",
    )
    agent = MAPPOAgent(env, mappo_cfg)

    print("MAPPO self-test")
    print("device:", agent.device)
    print("num_agents:", agent.num_agents)
    print("obs_dim:", agent.obs_dim)
    print("state_dim:", agent.state_dim)
    print("action_dim:", agent.action_dim)

    metrics = agent.train_one_update()
    print("\nMetrics after one update:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
