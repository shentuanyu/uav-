"""
Step 4 revised: 2D multi-UAV environment using a pluggable online target assignment module.

Completed so far:
- Step 1: reset() / step() multi-UAV kinematics.
- Step 2: circular obstacles, workspace boundary check, obstacle collision check,
          inter-UAV collision check, and simple rendering utilities.
- Step 3: static target generation, arrival judgment, success termination,
          and basic target-aware reward.
- Step 4: online target assignment through a replaceable assigner object.

Required extra file:
    assignment/target_assignment.py

Still NOT included in this step:
- Dynamic target movement / swapping.
- LiDAR observation.
- Full paper-style observation [z_i, u_i, g_i, q_i].
- Full DA-MAPPO reward function.

This file can be saved as:
    envs/uav_env.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from target_assignment import BaseTargetAssigner, build_assigner
except ImportError as exc:
    raise ImportError(
        "Could not import assignment.target_assignment. "
        "Please save the pluggable assignment module as assignment/target_assignment.py "
        "and make sure the project root is on PYTHONPATH."
    ) from exc


@dataclass
class UAVEnvConfig:
    """Configuration for the 2D UAV environment."""

    num_agents: int = 3
    world_size: float = 20.0          # World is [-world_size/2, world_size/2]^2
    dt: float = 0.1                   # Simulation time step
    max_steps: int = 600

    # Action bounds, consistent with the paper's simplified action setting.
    max_linear_velocity: float = 1.0  # m/s
    max_angular_velocity: float = 1.0 # rad/s

    # Initial UAV layout
    start_x: float = -8.0
    start_y_gap: float = 3.0

    # Static target settings
    num_targets: int = 3
    target_x: float = 8.0
    target_y_gap: float = 4.0
    arrival_threshold: float = 0.5

    # Assignment settings.
    # Options currently supported by build_assigner():
    #   "hungarian", "greedy", "fixed"
    assigner_name: str = "hungarian"

    # Obstacle settings
    num_obstacles: int = 20
    obstacle_radius_min: float = 0.25
    obstacle_radius_max: float = 0.25
    obstacle_area_x_min: float = -4.0
    obstacle_area_x_max: float = 5.0
    obstacle_area_y_min: float = -8.0
    obstacle_area_y_max: float = 8.0
    min_obstacle_spacing: float = 0.8

    # Collision / safety settings
    uav_radius: float = 0.20
    obstacle_safety_margin: float = 0.05
    inter_agent_min_distance: float = 0.45

    # Placeholder reward settings for Step 4.
    progress_reward_scale: float = 5.0
    arrival_reward: float = 100.0
    step_penalty: float = -1.0
    failure_penalty: float = -100.0

    seed: Optional[int] = None


class MultiUAV2DEnv:
    """
    Multi-UAV 2D environment with obstacles, static targets, and pluggable online assignment.

    State per UAV:
        x, y, heading, linear_velocity, angular_velocity

    Action per UAV:
        [v, omega]
        v     : linear velocity, clipped to [-max_linear_velocity, max_linear_velocity]
        omega : angular velocity, clipped to [-max_angular_velocity, max_angular_velocity]

    Observation per UAV in Step 4:
        [x_norm, y_norm, cos(heading), sin(heading), v_norm, omega_norm,
         target_dx_norm, target_dy_norm, target_distance_norm, target_bearing_norm]

    The target-related part is always computed from the currently assigned target.
    The assignment method is replaceable through self.target_assigner.
    """

    def __init__(
        self,
        config: Optional[UAVEnvConfig] = None,
        target_assigner: Optional[BaseTargetAssigner] = None,
    ):
        self.cfg = config or UAVEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        if self.cfg.num_targets < self.cfg.num_agents:
            raise ValueError("This environment expects num_targets >= num_agents.")

        self.num_agents = self.cfg.num_agents
        self.num_targets = self.cfg.num_targets
        self.action_dim = 2
        self.obs_dim = 10

        # Pluggable assignment module.
        self.target_assigner = target_assigner or build_assigner(self.cfg.assigner_name)
        self.assignment_info: Dict = {}

        self.positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.headings = np.zeros((self.num_agents,), dtype=np.float32)
        self.linear_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.angular_velocities = np.zeros((self.num_agents,), dtype=np.float32)

        self.target_positions = np.zeros((self.num_targets, 2), dtype=np.float32)
        self.assignments = np.arange(self.num_agents, dtype=np.int64)
        self.assignment_cost_matrix = np.zeros((self.num_agents, self.num_targets), dtype=np.float32)

        self.obstacle_centers = np.zeros((self.cfg.num_obstacles, 2), dtype=np.float32)
        self.obstacle_radii = np.zeros((self.cfg.num_obstacles,), dtype=np.float32)

        self.step_count = 0
        self.trajectory_lengths = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_target_distances = np.zeros((self.num_agents,), dtype=np.float32)
        self.arrived = np.zeros((self.num_agents,), dtype=bool)

        self.done = False
        self.termination_reason = ""

    def set_target_assigner(self, target_assigner: BaseTargetAssigner) -> None:
        """
        Replace the online target assignment module at runtime.

        Example:
            from assignment.target_assignment import GreedyNearestTargetAssigner
            env.set_target_assigner(GreedyNearestTargetAssigner())
        """
        self.target_assigner = target_assigner
        self._update_assignments()
        self.previous_target_distances = self._compute_assigned_target_distances()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment.

        Returns:
            obs: np.ndarray with shape [num_agents, obs_dim]
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.trajectory_lengths[:] = 0.0
        self.arrived[:] = False
        self.done = False
        self.termination_reason = ""

        self._reset_uavs()
        self._reset_targets()
        self._generate_obstacles()
        self._update_assignments()

        self.previous_target_distances = self._compute_assigned_target_distances()

        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Advance the simulation by one time step.

        Assignment order:
        1. reset() computes initial assignment for the initial observation.
        2. The policy acts based on that assignment-augmented observation.
        3. step(actions) applies actions and updates the environment.
        4. The assignment module is called again for the next observation.
        """
        if self.done:
            return (
                self._get_obs(),
                np.zeros((self.num_agents,), dtype=np.float32),
                np.ones((self.num_agents,), dtype=bool),
                self._get_info(),
            )

        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f"Expected actions with shape {(self.num_agents, self.action_dim)}, "
                f"but got {actions.shape}."
            )

        prev_positions = self.positions.copy()
        previous_distances = self.previous_target_distances.copy()

        self._apply_actions(actions)

        displacement = np.linalg.norm(self.positions - prev_positions, axis=1)
        self.trajectory_lengths += displacement.astype(np.float32)

        self.step_count += 1

        # Recompute assignment after the state transition.
        self._update_assignments()

        current_distances = self._compute_assigned_target_distances()
        newly_arrived = (current_distances <= self.cfg.arrival_threshold) & (~self.arrived)
        self.arrived |= current_distances <= self.cfg.arrival_threshold

        all_arrived = bool(np.all(self.arrived))
        boundary_violation = self._check_boundary_violation()
        obstacle_collision = self._check_obstacle_collision()
        inter_agent_collision = self._check_inter_agent_collision()
        timeout = self.step_count >= self.cfg.max_steps

        failure = bool(boundary_violation or obstacle_collision or inter_agent_collision)
        self.done = bool(all_arrived or failure or timeout)
        self.termination_reason = self._build_termination_reason(
            all_arrived=all_arrived,
            boundary_violation=boundary_violation,
            obstacle_collision=obstacle_collision,
            inter_agent_collision=inter_agent_collision,
            timeout=timeout,
        )

        rewards = self._compute_step4_rewards(
            previous_distances=previous_distances,
            current_distances=current_distances,
            newly_arrived=newly_arrived,
            failure=failure,
            all_arrived=all_arrived,
        )

        self.previous_target_distances = current_distances

        dones = np.full((self.num_agents,), self.done, dtype=bool)
        info = self._get_info(
            all_arrived=all_arrived,
            boundary_violation=boundary_violation,
            obstacle_collision=obstacle_collision,
            inter_agent_collision=inter_agent_collision,
            timeout=timeout,
            newly_arrived=newly_arrived.copy(),
            current_target_distances=current_distances.copy(),
        )

        return self._get_obs(), rewards, dones, info

    def _apply_actions(self, actions: np.ndarray) -> None:
        """Apply clipped velocity commands using unicycle-style 2D kinematics."""
        v_cmd = np.clip(
            actions[:, 0],
            -self.cfg.max_linear_velocity,
            self.cfg.max_linear_velocity,
        )
        omega_cmd = np.clip(
            actions[:, 1],
            -self.cfg.max_angular_velocity,
            self.cfg.max_angular_velocity,
        )

        self.linear_velocities = v_cmd.astype(np.float32)
        self.angular_velocities = omega_cmd.astype(np.float32)

        self.headings = self._wrap_angle(self.headings + self.angular_velocities * self.cfg.dt)
        self.positions[:, 0] += self.linear_velocities * np.cos(self.headings) * self.cfg.dt
        self.positions[:, 1] += self.linear_velocities * np.sin(self.headings) * self.cfg.dt

    def _reset_uavs(self) -> None:
        """Place UAVs vertically along the left side of the map."""
        center = (self.num_agents - 1) / 2.0
        for i in range(self.num_agents):
            self.positions[i, 0] = self.cfg.start_x
            self.positions[i, 1] = (i - center) * self.cfg.start_y_gap
            self.headings[i] = 0.0

        self.linear_velocities[:] = 0.0
        self.angular_velocities[:] = 0.0

    def _reset_targets(self) -> None:
        """Place targets vertically along the right side of the map."""
        center = (self.num_targets - 1) / 2.0
        for j in range(self.num_targets):
            self.target_positions[j, 0] = self.cfg.target_x
            self.target_positions[j, 1] = (j - center) * self.cfg.target_y_gap

    def _update_assignments(self) -> None:
        """
        Update agent-target assignment through the pluggable assignment module.
        """
        assignments, cost_matrix, assign_info = self.target_assigner.assign(
            agent_positions=self.positions.copy(),
            target_positions=self.target_positions.copy(),
            step_count=self.step_count,
            arrived=self.arrived.copy(),
        )

        assignments = np.asarray(assignments, dtype=np.int64)
        cost_matrix = np.asarray(cost_matrix, dtype=np.float32)

        if assignments.shape != (self.num_agents,):
            raise ValueError(
                f"Assigner returned assignments with shape {assignments.shape}, "
                f"expected {(self.num_agents,)}."
            )
        if np.any(assignments < 0) or np.any(assignments >= self.num_targets):
            raise ValueError(f"Invalid assignment indices: {assignments}.")

        self.assignments = assignments
        self.assignment_cost_matrix = cost_matrix
        self.assignment_info = dict(assign_info)

    def _generate_obstacles(self) -> None:
        """Generate non-overlapping circular obstacles inside the configured obstacle area."""
        if self.cfg.num_obstacles == 0:
            self.obstacle_centers = np.zeros((0, 2), dtype=np.float32)
            self.obstacle_radii = np.zeros((0,), dtype=np.float32)
            return

        centers = []
        radii = []
        max_attempts = 10_000
        attempts = 0

        while len(centers) < self.cfg.num_obstacles and attempts < max_attempts:
            attempts += 1
            radius = self.rng.uniform(self.cfg.obstacle_radius_min, self.cfg.obstacle_radius_max)
            center = np.array(
                [
                    self.rng.uniform(self.cfg.obstacle_area_x_min, self.cfg.obstacle_area_x_max),
                    self.rng.uniform(self.cfg.obstacle_area_y_min, self.cfg.obstacle_area_y_max),
                ],
                dtype=np.float32,
            )

            if self._is_valid_obstacle(center, radius, centers, radii):
                centers.append(center)
                radii.append(radius)

        if len(centers) < self.cfg.num_obstacles:
            raise RuntimeError(
                f"Only generated {len(centers)} obstacles out of {self.cfg.num_obstacles}. "
                "Try reducing num_obstacles or min_obstacle_spacing."
            )

        self.obstacle_centers = np.asarray(centers, dtype=np.float32)
        self.obstacle_radii = np.asarray(radii, dtype=np.float32)

    def _is_valid_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        existing_centers: list[np.ndarray],
        existing_radii: list[float],
    ) -> bool:
        """Check whether a newly sampled obstacle is valid."""
        for pos in self.positions:
            if np.linalg.norm(center - pos) < radius + self.cfg.uav_radius + self.cfg.min_obstacle_spacing:
                return False

        for target in self.target_positions:
            if np.linalg.norm(center - target) < radius + self.cfg.arrival_threshold + self.cfg.min_obstacle_spacing:
                return False

        for other_center, other_radius in zip(existing_centers, existing_radii):
            min_dist = radius + other_radius + self.cfg.min_obstacle_spacing
            if np.linalg.norm(center - other_center) < min_dist:
                return False

        return True

    def _compute_assigned_target_distances(self) -> np.ndarray:
        """Compute each UAV's distance to its currently assigned target."""
        assigned_targets = self.target_positions[self.assignments]
        return np.linalg.norm(self.positions - assigned_targets, axis=1).astype(np.float32)

    def _compute_assigned_target_vectors(self) -> np.ndarray:
        """Compute vector from each UAV to its currently assigned target."""
        assigned_targets = self.target_positions[self.assignments]
        return (assigned_targets - self.positions).astype(np.float32)

    def _compute_step4_rewards(
        self,
        previous_distances: np.ndarray,
        current_distances: np.ndarray,
        newly_arrived: np.ndarray,
        failure: bool,
        all_arrived: bool,
    ) -> np.ndarray:
        """
        Simple target-aware placeholder reward for Step 4.

        Note:
            Because assignment can change between two consecutive states,
            previous_distances and current_distances may correspond to different
            target identities. This is acceptable for this intermediate step.
            The full paper-style reward will be cleaned up in a later step.
        """
        progress = previous_distances - current_distances
        rewards = self.cfg.progress_reward_scale * progress + self.cfg.step_penalty

        rewards[newly_arrived] += self.cfg.arrival_reward

        if failure:
            rewards[:] = self.cfg.failure_penalty
        elif all_arrived:
            rewards[:] += self.cfg.arrival_reward

        return rewards.astype(np.float32)

    def _check_boundary_violation(self) -> bool:
        """Return True if any UAV leaves the square workspace."""
        half_size = self.cfg.world_size / 2.0
        return bool(np.any(np.abs(self.positions) > half_size))

    def _check_obstacle_collision(self) -> bool:
        """Return True if any UAV collides with any circular obstacle."""
        if self.cfg.num_obstacles == 0:
            return False

        diff = self.positions[:, None, :] - self.obstacle_centers[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        collision_thresholds = (
            self.cfg.uav_radius + self.obstacle_radii + self.cfg.obstacle_safety_margin
        )
        return bool(np.any(distances <= collision_thresholds[None, :]))

    def _check_inter_agent_collision(self) -> bool:
        """Return True if any pair of UAVs is too close."""
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist <= self.cfg.inter_agent_min_distance:
                    return True
        return False

    def _build_termination_reason(
        self,
        all_arrived: bool,
        boundary_violation: bool,
        obstacle_collision: bool,
        inter_agent_collision: bool,
        timeout: bool,
    ) -> str:
        reasons = []
        if all_arrived:
            reasons.append("success_all_arrived")
        if boundary_violation:
            reasons.append("boundary_violation")
        if obstacle_collision:
            reasons.append("obstacle_collision")
        if inter_agent_collision:
            reasons.append("inter_agent_collision")
        if timeout:
            reasons.append("timeout")
        return "+".join(reasons) if reasons else ""

    def _get_obs(self) -> np.ndarray:
        """
        Build Step-4 observations.

        Observation per UAV:
            x_norm, y_norm, cos(heading), sin(heading), v_norm, omega_norm,
            target_dx_norm, target_dy_norm, target_distance_norm, target_bearing_norm
        """
        half_size = self.cfg.world_size / 2.0
        max_distance = np.sqrt(2.0) * self.cfg.world_size

        x_norm = self.positions[:, 0] / half_size
        y_norm = self.positions[:, 1] / half_size
        cos_h = np.cos(self.headings)
        sin_h = np.sin(self.headings)
        v_norm = self.linear_velocities / self.cfg.max_linear_velocity
        omega_norm = self.angular_velocities / self.cfg.max_angular_velocity

        target_vectors = self._compute_assigned_target_vectors()
        target_dx_norm = target_vectors[:, 0] / self.cfg.world_size
        target_dy_norm = target_vectors[:, 1] / self.cfg.world_size
        target_distances = np.linalg.norm(target_vectors, axis=1)
        target_distance_norm = target_distances / max_distance

        target_angles = np.arctan2(target_vectors[:, 1], target_vectors[:, 0])
        relative_bearing = self._wrap_angle(target_angles - self.headings)
        target_bearing_norm = relative_bearing / np.pi

        obs = np.stack(
            [
                x_norm,
                y_norm,
                cos_h,
                sin_h,
                v_norm,
                omega_norm,
                target_dx_norm,
                target_dy_norm,
                target_distance_norm,
                target_bearing_norm,
            ],
            axis=1,
        ).astype(np.float32)

        return obs

    def get_global_state(self) -> np.ndarray:
        """
        Return a simple global state vector for the future centralized critic.
        """
        state_parts = [
            self.positions.reshape(-1) / (self.cfg.world_size / 2.0),
            np.cos(self.headings),
            np.sin(self.headings),
            self.linear_velocities / self.cfg.max_linear_velocity,
            self.angular_velocities / self.cfg.max_angular_velocity,
            self.target_positions.reshape(-1) / (self.cfg.world_size / 2.0),
            self.assignments.astype(np.float32) / max(1, self.num_targets - 1),
        ]
        return np.concatenate(state_parts).astype(np.float32)

    def _get_info(self, **extra_flags) -> Dict:
        """Return diagnostic information."""
        info = {
            "step_count": self.step_count,
            "done": self.done,
            "termination_reason": self.termination_reason,
            "positions": self.positions.copy(),
            "headings": self.headings.copy(),
            "trajectory_lengths": self.trajectory_lengths.copy(),
            "target_positions": self.target_positions.copy(),
            "assignments": self.assignments.copy(),
            "assignment_cost_matrix": self.assignment_cost_matrix.copy(),
            "assignment_info": dict(self.assignment_info),
            "arrived": self.arrived.copy(),
            "obstacle_centers": self.obstacle_centers.copy(),
            "obstacle_radii": self.obstacle_radii.copy(),
        }
        info.update(extra_flags)
        return info

    @staticmethod
    def _wrap_angle(angle: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi]."""
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    def render_state(self) -> None:
        """Simple text rendering for quick debugging."""
        print(f"Step: {self.step_count}")
        print(f"Done: {self.done}, reason: {self.termination_reason}")
        print(f"Assignments: {self.assignments.tolist()}")
        print(f"Assignment info: {self.assignment_info}")
        for i in range(self.num_agents):
            x, y = self.positions[i]
            target_id = self.assignments[i]
            tx, ty = self.target_positions[target_id]
            dist = np.linalg.norm(self.positions[i] - self.target_positions[target_id])
            print(
                f"UAV {i}: x={x:.2f}, y={y:.2f}, "
                f"heading={self.headings[i]:.2f}, "
                f"target={target_id}({tx:.2f},{ty:.2f}), "
                f"dist={dist:.2f}, arrived={self.arrived[i]}"
            )

    def render_matplotlib(self, ax=None):
        """
        Optional quick visualization using matplotlib.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        half_size = self.cfg.world_size / 2.0
        ax.set_xlim(-half_size, half_size)
        ax.set_ylim(-half_size, half_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"2D Multi-UAV Env | step={self.step_count}")

        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            circle = plt.Circle(center, radius, fill=True, alpha=0.5)
            ax.add_patch(circle)

        ax.scatter(
            self.target_positions[:, 0],
            self.target_positions[:, 1],
            marker="*",
            s=160,
            label="Targets",
        )

        ax.scatter(self.positions[:, 0], self.positions[:, 1], marker="o", label="UAVs")
        dx = np.cos(self.headings) * 0.5
        dy = np.sin(self.headings) * 0.5
        ax.quiver(self.positions[:, 0], self.positions[:, 1], dx, dy, angles="xy", scale_units="xy", scale=1)

        for i in range(self.num_agents):
            target = self.target_positions[self.assignments[i]]
            ax.plot(
                [self.positions[i, 0], target[0]],
                [self.positions[i, 1], target[1]],
                linestyle="--",
                linewidth=1,
            )
            ax.text(self.positions[i, 0], self.positions[i, 1], f"U{i}")
            ax.text(target[0], target[1], f"T{self.assignments[i]}")

        ax.legend(loc="upper left")
        ax.grid(True)
        return ax


if __name__ == "__main__":
    from target_assignment import GreedyNearestTargetAssigner

    cfg = UAVEnvConfig(
        num_agents=3,
        num_targets=3,
        num_obstacles=20,
        assigner_name="hungarian",
        seed=42,
    )
    env = MultiUAV2DEnv(cfg)

    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Global state shape:", env.get_global_state().shape)
    env.render_state()
    print("Targets:\n", env.target_positions)
    print("Initial assignment cost matrix:\n", env.assignment_cost_matrix)

    # Example of replacing the assignment module at runtime.
    # env.set_target_assigner(GreedyNearestTargetAssigner())
    # print("\nAfter replacing assigner with GreedyNearestTargetAssigner:")
    env.render_state()

    for _ in range(20):
        actions = np.array(
            [
                [1.0, 0.1],
                [1.0, 0.0],
                [1.0, -0.1],
            ],
            dtype=np.float32,
        )
        obs, rewards, dones, info = env.step(actions)
        env.render_state()
        print("Rewards:", rewards, "Dones:", dones, "Reason:", info["termination_reason"])
        print("Assignment:", info["assignments"])
        print("Assignment info:", info["assignment_info"])
        if dones.all():
            break

    # Uncomment for visualization test:
    import matplotlib.pyplot as plt
    env.render_matplotlib()
    plt.show()
