"""
Step 8: Rollout buffer for MAPPO / DA-MAPPO.

Save as:
    algorithms/rollout_buffer.py

This buffer stores one rollout collected from the multi-UAV environment:
- local observations for each agent
- centralized global states
- actions
- log probabilities from the old policy
- rewards
- done masks
- centralized critic values

It also computes:
- GAE advantages
- discounted returns

Expected tensor shapes during storage:
    obs:        [num_agents, obs_dim]
    state:      [state_dim]
    actions:    [num_agents, action_dim]
    log_probs:  [num_agents]
    rewards:    [num_agents]
    dones:      [num_agents] or scalar bool
    value:      scalar, centralized V(s_t)

After compute_returns_and_advantages():
    advantages: [rollout_steps, num_agents]
    returns:    [rollout_steps, num_agents]

For actor update, each agent-time pair is treated as one sample.
For critic update, the same centralized state is repeated per agent so that
critic loss can be trained against per-agent returns. This is a simple and
common implementation choice for early MAPPO reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

@dataclass
class BufferConfig:
    rollout_steps: int
    num_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class RolloutBuffer:
    """Fixed-size rollout buffer for MAPPO."""

    def __init__(self, config: BufferConfig):
        self.cfg = config
        self.rollout_steps = config.rollout_steps
        self.num_agents = config.num_agents
        self.obs_dim = config.obs_dim
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = torch.device(config.device)

        self.obs = np.zeros(
            (self.rollout_steps, self.num_agents, self.obs_dim), dtype=np.float32
        )
        self.states = np.zeros((self.rollout_steps, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(
            (self.rollout_steps, self.num_agents, self.action_dim), dtype=np.float32
        )
        self.log_probs = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps,), dtype=np.float32)

        self.advantages = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        """Clear pointer. Existing arrays are overwritten during storage."""
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        value: float,
    ) -> None:
        """Store one environment transition."""
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer is full. Call reset() before adding more data.")

        obs = np.asarray(obs, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        log_probs = np.asarray(log_probs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if obs.shape != (self.num_agents, self.obs_dim):
            raise ValueError(f"obs shape {obs.shape}, expected {(self.num_agents, self.obs_dim)}")
        if state.shape != (self.state_dim,):
            raise ValueError(f"state shape {state.shape}, expected {(self.state_dim,)}")
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f"actions shape {actions.shape}, expected {(self.num_agents, self.action_dim)}"
            )
        if log_probs.shape != (self.num_agents,):
            raise ValueError(f"log_probs shape {log_probs.shape}, expected {(self.num_agents,)}")
        if rewards.shape != (self.num_agents,):
            raise ValueError(f"rewards shape {rewards.shape}, expected {(self.num_agents,)}")

        # Allow scalar done and expand it to all agents.
        if dones.shape == ():
            dones = np.full((self.num_agents,), float(dones), dtype=np.float32)
        if dones.shape != (self.num_agents,):
            raise ValueError(f"dones shape {dones.shape}, expected {(self.num_agents,)}")

        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = float(value)

        self.ptr += 1
        self.full = self.ptr == self.rollout_steps

    # def compute_returns_and_advantages(
    #     self,
    #     last_value: float,
    #     last_done: bool | np.ndarray,
    # ) -> None:
    #     """
    #     Compute GAE advantages and returns.
    #
    #     Args:
    #         last_value:
    #             V(s_{T}) for the state after the last stored transition.
    #             Use 0 if the rollout ended with a terminal state.
    #
    #         last_done:
    #             Done flag for the state after the last stored transition.
    #             If True, bootstrap value is masked out.
    #     """
    #     valid_steps = self.ptr
    #     if valid_steps == 0:
    #         raise RuntimeError("Cannot compute returns on an empty buffer.")
    #
    #     if isinstance(last_done, np.ndarray):
    #         last_done_float = float(np.all(last_done))
    #     else:
    #         last_done_float = float(last_done)
    #
    #     gae = np.zeros((self.num_agents,), dtype=np.float32)
    #
    #     for step in reversed(range(valid_steps)):
    #         if step == valid_steps - 1:
    #             next_value = float(last_value)
    #             next_nonterminal = 1.0 - last_done_float
    #         else:
    #             next_value = self.values[step + 1]
    #             # In this environment all agents share termination, but using all() is robust.
    #             next_nonterminal = 1.0 - float(np.all(self.dones[step + 1]))
    #
    #         delta = (
    #             self.rewards[step]
    #             + self.cfg.gamma * next_value * next_nonterminal
    #             - self.values[step]
    #         )
    #         gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * gae
    #         self.advantages[step] = gae
    #         self.returns[step] = gae + self.values[step]
    def compute_returns_and_advantages(
            self,
            last_value: float,
            last_done: bool | np.ndarray,
    ) -> None:
        valid_steps = self.ptr
        if valid_steps == 0:
            raise RuntimeError("Cannot compute returns on an empty buffer.")

        if isinstance(last_done, np.ndarray):
            last_done_float = float(np.all(last_done))
        else:
            last_done_float = float(last_done)

        gae = np.zeros((self.num_agents,), dtype=np.float32)

        for step in reversed(range(valid_steps)):
            done_t = float(np.all(self.dones[step]))

            if step == valid_steps - 1:
                next_value = float(last_value)
                next_nonterminal = 1.0 - last_done_float
            else:
                next_value = self.values[step + 1]
                next_nonterminal = 1.0 - done_t

            delta = (
                    self.rewards[step]
                    + self.cfg.gamma * next_value * next_nonterminal
                    - self.values[step]
            )

            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * gae

            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]


    def get_training_tensors(self, normalize_advantages: bool = True) -> Dict[str, torch.Tensor]:
        """
        Flatten rollout into tensors for PPO update.

        Returns shapes:
            obs:          [T * N, obs_dim]
            states:       [T * N, state_dim]
            actions:      [T * N, action_dim]
            old_log_probs:[T * N]
            advantages:   [T * N]
            returns:      [T * N]
            old_values:   [T * N]
        """
        valid_steps = self.ptr
        if valid_steps == 0:
            raise RuntimeError("Buffer is empty.")

        obs = self.obs[:valid_steps].reshape(valid_steps * self.num_agents, self.obs_dim)
        states = np.repeat(self.states[:valid_steps], self.num_agents, axis=0)
        actions = self.actions[:valid_steps].reshape(valid_steps * self.num_agents, self.action_dim)
        old_log_probs = self.log_probs[:valid_steps].reshape(valid_steps * self.num_agents)
        advantages = self.advantages[:valid_steps].reshape(valid_steps * self.num_agents)
        returns = self.returns[:valid_steps].reshape(valid_steps * self.num_agents)
        old_values = np.repeat(self.values[:valid_steps], self.num_agents, axis=0)

        if normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            "states": torch.as_tensor(states, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            "old_log_probs": torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            "old_values": torch.as_tensor(old_values, dtype=torch.float32, device=self.device),
        }

    def iter_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
        normalize_advantages: bool = True,
    ):
        """
        Yield minibatches for PPO update.

        If batch_size >= total samples, this yields one full batch.
        """
        tensors = self.get_training_tensors(normalize_advantages=normalize_advantages)
        total = tensors["obs"].shape[0]

        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
            yield {key: value[batch_idx_t] for key, value in tensors.items()}

    def summary(self) -> Dict[str, float]:
        """Return simple diagnostics for debugging."""
        valid_steps = self.ptr
        if valid_steps == 0:
            return {"valid_steps": 0.0}

        return {
            "valid_steps": float(valid_steps),
            "mean_reward": float(self.rewards[:valid_steps].mean()),
            "mean_return": float(self.returns[:valid_steps].mean()),
            "mean_advantage": float(self.advantages[:valid_steps].mean()),
            "done_fraction": float(self.dones[:valid_steps].mean()),
        }


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = BufferConfig(
        rollout_steps=8,
        num_agents=3,
        obs_dim=45,
        state_dim=33,
        action_dim=2,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )
    buffer = RolloutBuffer(cfg)

    for t in range(cfg.rollout_steps):
        obs = np.random.randn(cfg.num_agents, cfg.obs_dim).astype(np.float32)
        state = np.random.randn(cfg.state_dim).astype(np.float32)
        actions = np.tanh(np.random.randn(cfg.num_agents, cfg.action_dim)).astype(np.float32)
        log_probs = np.random.randn(cfg.num_agents).astype(np.float32)
        rewards = np.random.randn(cfg.num_agents).astype(np.float32)
        dones = np.zeros(cfg.num_agents, dtype=np.float32)
        value = np.random.randn()
        buffer.add(obs, state, actions, log_probs, rewards, dones, value)

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=False)
    tensors = buffer.get_training_tensors()

    print("RolloutBuffer self-test")
    print("summary:", buffer.summary())
    for key, value in tensors.items():
        print(f"{key}: {tuple(value.shape)}")

    print("\nMinibatches:")
    for batch in buffer.iter_minibatches(batch_size=10):
        print({key: tuple(value.shape) for key, value in batch.items()})
