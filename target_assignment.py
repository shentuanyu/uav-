"""
Pluggable online target assignment module for DA-MAPPO reproduction.

Save as:
    assignment/target_assignment.py

Design goal:
- The environment should not depend on a specific assignment algorithm.
- Any online assignment method only needs to implement BaseTargetAssigner.assign().
- Hungarian assignment is provided as the default method.
- Later you can replace it with auction-based assignment, greedy assignment,
  learned assignment, graph matching, attention-based assignment, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "HungarianTargetAssigner requires scipy. Install it with: pip install scipy"
    ) from exc


AssignmentResult = Tuple[np.ndarray, np.ndarray, Dict]


class BaseTargetAssigner(ABC):
    """
    Abstract base class for online target assignment.

    Required input:
        agent_positions: np.ndarray, shape [num_agents, 2]
        target_positions: np.ndarray, shape [num_targets, 2]

    Required output:
        assignments: np.ndarray, shape [num_agents]
            assignments[i] = target index assigned to agent i

        cost_matrix: np.ndarray, shape [num_agents, num_targets]
            cost matrix used by the assigner. If a method does not explicitly
            use a cost matrix, it can still return a diagnostic pseudo-cost.

        info: dict
            extra diagnostic information.
    """

    @abstractmethod
    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        raise NotImplementedError


@dataclass
class HungarianTargetAssigner(BaseTargetAssigner):
    """
    Minimum-cost one-to-one assignment using Hungarian algorithm.

    This corresponds to the assignment module used in the paper:
        C[i, j] = ||p_i - q_j||_2^2

    Args:
        squared_distance: whether to use squared Euclidean distance.
                          The paper uses squared Euclidean distance.
    """

    squared_distance: bool = True

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        self._validate_inputs(agent_positions, target_positions)

        cost_matrix = self.build_cost_matrix(agent_positions, target_positions)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        num_agents = agent_positions.shape[0]
        assignments = np.full((num_agents,), -1, dtype=np.int64)
        for agent_id, target_id in zip(row_ind, col_ind):
            assignments[agent_id] = target_id

        if np.any(assignments < 0):
            raise RuntimeError(
                "Hungarian assignment failed to assign every agent. "
                "Check that num_targets >= num_agents."
            )

        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        info = {
            "assigner": "hungarian",
            "total_assignment_cost": total_cost,
        }
        return assignments, cost_matrix.astype(np.float32), info

    def build_cost_matrix(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        distances_sq = np.sum(diff ** 2, axis=-1)
        if self.squared_distance:
            return distances_sq.astype(np.float32)
        return np.sqrt(distances_sq + 1e-8).astype(np.float32)

    @staticmethod
    def _validate_inputs(
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> None:
        if agent_positions.ndim != 2 or agent_positions.shape[1] != 2:
            raise ValueError(
                f"agent_positions must have shape [num_agents, 2], got {agent_positions.shape}."
            )
        if target_positions.ndim != 2 or target_positions.shape[1] != 2:
            raise ValueError(
                f"target_positions must have shape [num_targets, 2], got {target_positions.shape}."
            )
        if target_positions.shape[0] < agent_positions.shape[0]:
            raise ValueError(
                "num_targets must be >= num_agents for one-to-one assignment. "
                f"Got num_targets={target_positions.shape[0]}, "
                f"num_agents={agent_positions.shape[0]}."
            )


@dataclass
class GreedyNearestTargetAssigner(BaseTargetAssigner):
    """
    Simple greedy nearest-target assigner.

    This is mainly useful as a quick replaceable baseline.
    It sorts all agent-target pairs by distance and greedily picks non-conflicting pairs.
    It is not globally optimal, but it is fast and easy to compare against Hungarian.
    """

    squared_distance: bool = True

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        distances_sq = np.sum(diff ** 2, axis=-1)
        cost_matrix = distances_sq if self.squared_distance else np.sqrt(distances_sq + 1e-8)

        num_agents, num_targets = cost_matrix.shape
        assignments = np.full((num_agents,), -1, dtype=np.int64)
        used_targets = set()

        pairs = [
            (float(cost_matrix[i, j]), i, j)
            for i in range(num_agents)
            for j in range(num_targets)
        ]
        pairs.sort(key=lambda x: x[0])

        for _, agent_id, target_id in pairs:
            if assignments[agent_id] != -1:
                continue
            if target_id in used_targets:
                continue
            assignments[agent_id] = target_id
            used_targets.add(target_id)
            if np.all(assignments >= 0):
                break

        if np.any(assignments < 0):
            raise RuntimeError("Greedy assignment failed to assign every agent.")

        total_cost = float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents)))
        info = {
            "assigner": "greedy_nearest",
            "total_assignment_cost": total_cost,
        }
        return assignments, cost_matrix.astype(np.float32), info


class FixedTargetAssigner(BaseTargetAssigner):
    """
    Fixed assignment baseline: agent i -> target i.

    Useful for ablation:
    - FixedTargetAssigner + MAPPO approximates ordinary MAPPO with fixed targets.
    - HungarianTargetAssigner + MAPPO is DA-MAPPO-style dynamic assignment.
    """

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        num_agents = agent_positions.shape[0]
        assignments = np.arange(num_agents, dtype=np.int64)

        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        cost_matrix = np.sum(diff ** 2, axis=-1).astype(np.float32)

        info = {
            "assigner": "fixed",
            "total_assignment_cost": float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents))),
        }
        return assignments, cost_matrix, info


def build_assigner(name: str, **kwargs) -> BaseTargetAssigner:
    """
    Factory function for convenient config-based construction.

    Example:
        assigner = build_assigner("hungarian")
        assigner = build_assigner("greedy")
        assigner = build_assigner("fixed")
    """
    name = name.lower()
    if name in {"hungarian", "hungarian_target", "min_cost"}:
        return HungarianTargetAssigner(**kwargs)
    if name in {"greedy", "greedy_nearest"}:
        return GreedyNearestTargetAssigner(**kwargs)
    if name in {"fixed", "identity"}:
        return FixedTargetAssigner()
    raise ValueError(f"Unknown assigner name: {name}")


if __name__ == "__main__":
    agents = np.array([
        [-8.0, -3.0],
        [-8.0, 0.0],
        [-8.0, 3.0],
    ], dtype=np.float32)

    targets = np.array([
        [8.0, -4.0],
        [8.0, 0.0],
        [8.0, 4.0],
    ], dtype=np.float32)

    for method in ["fixed", "greedy", "hungarian"]:
        assigner = build_assigner(method)
        assignments, cost_matrix, info = assigner.assign(agents, targets)
        print(f"\nMethod: {method}")
        print("Assignments:", assignments)
        print("Cost matrix:\n", cost_matrix)
        print("Info:", info)
