"""
Step 7: Actor-Critic network modules for MAPPO / DA-MAPPO.

Save as:
    algorithms/actor_critic.py

This file implements:
- Shared Gaussian actor for all UAV agents.
- Centralized critic for MAPPO-style CTDE training.
- MLP builder utility.
- Orthogonal initialization utility.
- Small self-test at the bottom.

Design:
- Actor input: local observation o_i = [z_i, u_i, g_i, q_i]
- Actor output: continuous action distribution over [v, omega]
- Critic input: global state s_t from env.get_global_state()
- Critic output: scalar V(s_t)

The environment action bound is [-1, 1] for both v and omega. Therefore,
the actor uses a tanh-squashed Gaussian distribution by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class NetworkConfig:
    """Network hyperparameters."""

    obs_dim: int
    state_dim: int
    action_dim: int = 2
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    activation: str = "tanh"
    log_std_init: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    use_orthogonal_init: bool = True


class TanhNormal:
    """
    Tanh-squashed Normal distribution.

    The actor samples raw_action from Normal(mean, std), then squashes it:
        action = tanh(raw_action)

    Since tanh maps real values to (-1, 1), this directly matches the current
    environment action bounds for [v, omega].

    log_prob uses the standard tanh correction:
        log pi(a) = log N(raw_action) - sum log(1 - tanh(raw_action)^2)
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.normal = Normal(mean, self.std)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return action, log_prob."""
        raw_action = self.normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = self.log_prob_from_raw(raw_action, action)
        return action, log_prob

    def deterministic(self) -> torch.Tensor:
        """Return deterministic action for evaluation."""
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Compute log_prob of a squashed action."""
        action = torch.clamp(action, -0.999999, 0.999999)
        raw_action = torch.atanh(action)
        return self.log_prob_from_raw(raw_action, action)

    def log_prob_from_raw(self, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute tanh-corrected log probability."""
        log_prob = self.normal.log_prob(raw_action)
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob - correction
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy.

        This returns the entropy of the pre-squash Gaussian. It is commonly used
        as a practical entropy bonus for tanh-squashed policies.
        """
        return self.normal.entropy().sum(dim=-1)


class GaussianActor(nn.Module):
    """
    Shared actor network for all UAV agents.

    Input:
        obs: shape [..., obs_dim]

    Output:
        mean and log_std for a tanh-squashed Gaussian action distribution.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.backbone = build_mlp(
            input_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), config.log_std_init))

        if config.use_orthogonal_init:
            orthogonal_init(self)
            nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> TanhNormal:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std,
            self.config.min_log_std,
            self.config.max_log_std,
        )
        log_std = log_std.expand_as(mean)
        return TanhNormal(mean, log_std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample or deterministically choose actions.

        Args:
            obs: shape [batch, obs_dim] or [num_agents, obs_dim]
            deterministic: if True, use tanh(mean)

        Returns:
            actions: shape [..., action_dim], in [-1, 1]
            log_probs: shape [...]
        """
        dist = self.forward(obs)
        if deterministic:
            actions = dist.deterministic()
            log_probs = dist.log_prob(actions)
        else:
            actions, log_probs = dist.sample()
        return actions, log_probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and entropy for PPO update.

        Args:
            obs: shape [batch, obs_dim]
            actions: shape [batch, action_dim]

        Returns:
            log_probs: shape [batch]
            entropy: shape [batch]
        """
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized critic network for MAPPO.

    Input:
        global_state: shape [..., state_dim]

    Output:
        V(s): shape [...]
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.value_net = build_mlp(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
        )

        if config.use_orthogonal_init:
            orthogonal_init(self)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        value = self.value_net(global_state)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    """
    Convenience wrapper holding actor and critic together.

    MAPPO will still optimize actor and critic losses separately, but this wrapper
    makes checkpointing and device movement easier.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.actor = GaussianActor(config)
        self.critic = CentralizedCritic(config)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor.act(obs, deterministic=deterministic)

    def value(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic(global_state)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        values = self.critic(global_state)
        return log_probs, entropy, values


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    activation: str = "tanh",
) -> nn.Sequential:
    """Build a simple feed-forward MLP."""
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1.")

    act = get_activation(activation)
    layers = []

    last_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(act())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def get_activation(name: str):
    """Return activation class by name."""
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to all linear layers."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)


def count_parameters(modules: Iterable[nn.Module]) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    num_agents = 3
    obs_dim = 45
    state_dim = 3 * 2 + 3 + 3 + 3 + 3 + 3 + 3 + 3 * 2 + 3
    action_dim = 2

    config = NetworkConfig(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_hidden_layers=3,
        activation="tanh",
    )

    model = ActorCritic(config)

    obs = torch.randn(num_agents, obs_dim)
    global_state = torch.randn(1, state_dim)

    actions, log_probs = model.act(obs, deterministic=False)
    values = model.value(global_state)
    eval_log_probs, entropy, _ = model.evaluate_actions(obs, global_state.repeat(num_agents, 1), actions)

    print("Actor-Critic self-test")
    print("obs shape:", obs.shape)
    print("global_state shape:", global_state.shape)
    print("actions shape:", actions.shape)
    print("actions range:", float(actions.min()), float(actions.max()))
    print("log_probs shape:", log_probs.shape)
    print("values shape:", values.shape)
    print("eval_log_probs shape:", eval_log_probs.shape)
    print("entropy shape:", entropy.shape)
    print("actor params:", count_parameters([model.actor]))
    print("critic params:", count_parameters([model.critic]))
