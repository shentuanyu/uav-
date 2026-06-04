"""
修改到达逻辑，uav到达后不动
Multi-UAV 2D environment with:
- Pluggable online target assignment
- Paper-style observation o_i = [z_i, u_i, g_i, q_i]
- 2D LiDAR obstacle sensing
- Step-6 trainable reward
- Arrival fix:
    1. UAVs freeze after arrival.
    2. Task success is judged by target_arrived, not only UAV_arrived.

Save as:
    envs/uav_env.py

Required extra file:
    assignment/target_assignment.py
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
    world_size: float = 20.0
    dt: float = 0.1
    max_steps: int = 600

    # Action bounds.
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 1.0

    # Initial UAV layout.
    start_x: float = -8.0
    start_y_gap: float = 3.0

    # Static target settings.
    num_targets: int = 3
    target_x: float = 8.0
    target_y_gap: float = 4.0
    arrival_threshold: float = 0.5

    # Assignment settings. Options: "hungarian", "greedy", "fixed".
    assigner_name: str = "hungarian"

    # LiDAR settings.
    lidar_num_rays: int = 35
    lidar_range: float = 5.0
    lidar_fov: float = 2.0 * np.pi

    # Communication / topology observation.
    communication_range: float = 10.0
    use_communication_range_mask: bool = False

    # Obstacle settings.
    num_obstacles: int = 20
    obstacle_radius_min: float = 0.2
    obstacle_radius_max: float = 0.2
    obstacle_area_x_min: float = -4.0
    obstacle_area_x_max: float = 5.0
    obstacle_area_y_min: float = -8.0
    obstacle_area_y_max: float = 8.0
    min_obstacle_spacing: float = 0.8

    # Collision / safety settings.
    uav_radius: float = 0.20
    obstacle_safety_margin: float = 0.05
    inter_agent_min_distance: float = 0.45

    # Step-6 trainable simple reward settings.
    reward_scale: float = 1.0
    progress_reward_scale: float = 5.0
    team_distance_scale: float = 0.05
    arrival_reward: float = 80.0
    all_arrived_bonus: float = 80.0
    step_penalty: float = -0.5
    failure_penalty: float = -200.0

    # Soft safety reward settings.
    lidar_warning_distance: float = 1.8
    lidar_danger_distance: float = 0.7
    lidar_warning_penalty_scale: float = 6.0
    lidar_danger_penalty: float = -30.0

    inter_agent_warning_distance: float = 1.0
    inter_agent_warning_penalty_scale: float = 2.0

    acceleration_penalty_scale_v: float = 0.03
    acceleration_penalty_scale_omega: float = 0.02

    seed: Optional[int] = None


class MultiUAV2DEnv:
    """
    Multi-UAV 2D environment with paper-style observations, pluggable assignment,
    simple trainable reward, and corrected arrival handling.

    Action per UAV:
        [v, omega]

    Observation per UAV:
        [z_i, u_i, g_i, q_i]

    Specifically:
        z_i: normalized LiDAR distances, shape [D]
        u_i: [v_norm, omega_norm, a_v_norm, a_omega_norm]
        g_i: [target_distance_norm, target_bearing_norm]
        q_i: relative teammate positions normalized by world_size,
             shape [2 * (num_agents - 1)]
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
        self.obs_dim = self.cfg.lidar_num_rays + 4 + 2 + 2 * (self.num_agents - 1)

        self.target_assigner = target_assigner or build_assigner(self.cfg.assigner_name)
        self.assignment_info: Dict = {}

        self.positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.headings = np.zeros((self.num_agents,), dtype=np.float32)
        self.linear_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.angular_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_linear_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_angular_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.linear_accelerations = np.zeros((self.num_agents,), dtype=np.float32)
        self.angular_accelerations = np.zeros((self.num_agents,), dtype=np.float32)

        self.target_positions = np.zeros((self.num_targets, 2), dtype=np.float32)
        self.assignments = np.arange(self.num_agents, dtype=np.int64)
        self.assignment_cost_matrix = np.zeros((self.num_agents, self.num_targets), dtype=np.float32)

        self.obstacle_centers = np.zeros((self.cfg.num_obstacles, 2), dtype=np.float32)
        self.obstacle_radii = np.zeros((self.cfg.num_obstacles,), dtype=np.float32)

        self.step_count = 0
        self.trajectory_lengths = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_target_distances = np.zeros((self.num_agents,), dtype=np.float32)

        # UAV-level arrival: whether each UAV has reached a target once.
        self.arrived = np.zeros((self.num_agents,), dtype=bool)

        # Target-level arrival: whether each target has been completed.
        # Success is now judged by this array.
        self.target_arrived = np.zeros((self.num_targets,), dtype=bool)

        self.done = False
        self.termination_reason = ""
        self.last_reward_terms: Dict[str, np.ndarray] = {}

    def set_target_assigner(self, target_assigner: BaseTargetAssigner) -> None:
        """Replace the online target assignment module at runtime."""
        self.target_assigner = target_assigner
        self._update_assignments()
        self.previous_target_distances = self._compute_assigned_target_distances()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return observations with shape [num_agents, obs_dim]."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.trajectory_lengths[:] = 0.0
        self.arrived[:] = False
        self.target_arrived[:] = False
        self.done = False
        self.termination_reason = ""
        self.last_reward_terms = {}

        self._reset_uavs()
        self._reset_targets()
        self._generate_obstacles()
        self._update_assignments()

        self.previous_target_distances = self._compute_assigned_target_distances()
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Advance the simulation by one time step."""
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

        self._update_assignments()

        current_distances = self._compute_assigned_target_distances()

        # UAV-level arrival: a UAV is newly arrived if it reaches its currently assigned target now.
        newly_arrived = (current_distances <= self.cfg.arrival_threshold) & (~self.arrived)

        # Target-level arrival: mark the actual assigned targets completed when UAVs newly arrive.
        newly_arrived_target_ids = self.assignments[newly_arrived]
        if len(newly_arrived_target_ids) > 0:
            self.target_arrived[newly_arrived_target_ids] = True

        # Freeze these UAVs in later steps.
        self.arrived |= current_distances <= self.cfg.arrival_threshold

        # Success is judged by whether every target is completed.
        # This is suitable for num_agents == num_targets.
        all_arrived = bool(np.all(self.target_arrived))

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

        rewards = self._compute_simple_trainable_rewards(
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
            newly_arrived_target_ids=newly_arrived_target_ids.copy(),
            current_target_distances=current_distances.copy(),
        )

        return self._get_obs(), rewards, dones, info

    def _apply_actions(self, actions: np.ndarray) -> None:
        """Apply clipped velocity commands using unicycle-style 2D kinematics."""
        self.previous_linear_velocities = self.linear_velocities.copy()
        self.previous_angular_velocities = self.angular_velocities.copy()

        v_cmd = np.clip(actions[:, 0], -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity)
        omega_cmd = np.clip(actions[:, 1], -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity)

        # Arrival fix 1:
        # UAVs that have already arrived are frozen from the next step onward.
        # This prevents post-arrival drift or circular trajectories around targets.
        v_cmd[self.arrived] = 0.0
        omega_cmd[self.arrived] = 0.0

        self.linear_velocities = v_cmd.astype(np.float32)
        self.angular_velocities = omega_cmd.astype(np.float32)

        self.linear_accelerations = (
            (self.linear_velocities - self.previous_linear_velocities) / self.cfg.dt
        ).astype(np.float32)
        self.angular_accelerations = (
            (self.angular_velocities - self.previous_angular_velocities) / self.cfg.dt
        ).astype(np.float32)

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
        self.previous_linear_velocities[:] = 0.0
        self.previous_angular_velocities[:] = 0.0
        self.linear_accelerations[:] = 0.0
        self.angular_accelerations[:] = 0.0

    def _reset_targets(self) -> None:
        """Place targets vertically along the right side of the map."""
        center = (self.num_targets - 1) / 2.0
        for j in range(self.num_targets):
            self.target_positions[j, 0] = self.cfg.target_x
            self.target_positions[j, 1] = (j - center) * self.cfg.target_y_gap

    def _update_assignments(self) -> None:
        """Update agent-target assignment through the pluggable assignment module."""
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

    # ---------------------------------------------------------------------
    # Observation construction: o_i = [z_i, u_i, g_i, q_i]
    # ---------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build paper-style observations with shape [num_agents, obs_dim]."""
        lidar_obs = self._compute_lidar_observations()
        ego_obs = self._compute_ego_motion_observations()
        target_obs = self._compute_target_observations()
        topology_obs = self._compute_topology_observations()

        obs = np.concatenate([lidar_obs, ego_obs, target_obs, topology_obs], axis=1)
        if obs.shape != (self.num_agents, self.obs_dim):
            raise RuntimeError(
                f"Observation shape mismatch: got {obs.shape}, "
                f"expected {(self.num_agents, self.obs_dim)}."
            )
        return obs.astype(np.float32)

    def _compute_lidar_observations(self) -> np.ndarray:
        """
        Simulate simple 2D LiDAR by ray-circle intersection.

        Output is normalized to [0, 1]:
            1.0 means no obstacle within lidar_range.
            smaller values mean closer obstacles.
        """
        num_rays = self.cfg.lidar_num_rays
        max_range = self.cfg.lidar_range
        half_fov = self.cfg.lidar_fov / 2.0
        relative_angles = np.linspace(-half_fov, half_fov, num_rays, dtype=np.float32)

        lidar = np.full((self.num_agents, num_rays), max_range, dtype=np.float32)
        if self.obstacle_centers.shape[0] == 0:
            return lidar / max_range

        for i in range(self.num_agents):
            origin = self.positions[i]
            ray_angles = self.headings[i] + relative_angles
            ray_dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1).astype(np.float32)

            for r_id, direction in enumerate(ray_dirs):
                closest = max_range
                for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
                    hit_distance = self._ray_circle_intersection_distance(
                        origin=origin,
                        direction=direction,
                        center=center,
                        radius=radius + self.cfg.uav_radius + self.cfg.obstacle_safety_margin,
                        max_range=max_range,
                    )
                    if hit_distance is not None and hit_distance < closest:
                        closest = hit_distance
                lidar[i, r_id] = closest

        return np.clip(lidar / max_range, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _ray_circle_intersection_distance(
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
        max_range: float,
    ) -> Optional[float]:
        """Return nearest positive intersection distance between a ray and a circle."""
        oc = origin - center
        b = 2.0 * float(np.dot(direction, oc))
        c = float(np.dot(oc, oc) - radius * radius)
        discriminant = b * b - 4.0 * c

        if discriminant < 0.0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        candidates = [t for t in (t1, t2) if 0.0 <= t <= max_range]
        if not candidates:
            return None
        return float(min(candidates))

    def _compute_ego_motion_observations(self) -> np.ndarray:
        """u_i = [v, omega, a_v, a_omega], normalized."""
        max_linear_acc = max(self.cfg.max_linear_velocity / self.cfg.dt, 1e-6)
        max_angular_acc = max(self.cfg.max_angular_velocity / self.cfg.dt, 1e-6)

        v_norm = self.linear_velocities / self.cfg.max_linear_velocity
        omega_norm = self.angular_velocities / self.cfg.max_angular_velocity
        a_v_norm = np.clip(self.linear_accelerations / max_linear_acc, -1.0, 1.0)
        a_omega_norm = np.clip(self.angular_accelerations / max_angular_acc, -1.0, 1.0)

        return np.stack([v_norm, omega_norm, a_v_norm, a_omega_norm], axis=1).astype(np.float32)

    def _compute_target_observations(self) -> np.ndarray:
        """g_i = [distance_to_assigned_target, relative_bearing], normalized."""
        max_distance = np.sqrt(2.0) * self.cfg.world_size
        target_vectors = self._compute_assigned_target_vectors()
        target_distances = np.linalg.norm(target_vectors, axis=1)
        distance_norm = target_distances / max_distance

        target_angles = np.arctan2(target_vectors[:, 1], target_vectors[:, 0])
        relative_bearing = self._wrap_angle(target_angles - self.headings)
        bearing_norm = relative_bearing / np.pi

        return np.stack([distance_norm, bearing_norm], axis=1).astype(np.float32)

    def _compute_topology_observations(self) -> np.ndarray:
        """q_i = normalized relative positions of all teammates."""
        topology = np.zeros((self.num_agents, 2 * (self.num_agents - 1)), dtype=np.float32)
        scale = self.cfg.world_size

        for i in range(self.num_agents):
            features = []
            for j in range(self.num_agents):
                if i == j:
                    continue
                rel = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(rel)
                if self.cfg.use_communication_range_mask and dist > self.cfg.communication_range:
                    rel = np.zeros(2, dtype=np.float32)
                features.extend((rel / scale).tolist())
            topology[i] = np.asarray(features, dtype=np.float32)

        return topology

    # ---------------------------------------------------------------------
    # Reward
    # ---------------------------------------------------------------------
    def _compute_simple_trainable_rewards(
        self,
        previous_distances: np.ndarray,
        current_distances: np.ndarray,
        newly_arrived: np.ndarray,
        failure: bool,
        all_arrived: bool,
    ) -> np.ndarray:
        """
        Compute a stable intermediate reward for early training.

        Main learning signals:
        - move toward assigned targets;
        - let the whole team get closer;
        - avoid obstacles and teammates;
        - keep actions smooth;
        - finish quickly.
        """
        progress_term = self.cfg.progress_reward_scale * (previous_distances - current_distances)
        team_distance_term = -self.cfg.team_distance_scale * np.sum(current_distances)
        team_distance_term = np.full((self.num_agents,), team_distance_term, dtype=np.float32)

        arrival_term = np.zeros((self.num_agents,), dtype=np.float32)
        arrival_term[newly_arrived] = self.cfg.arrival_reward
        if all_arrived:
            arrival_term += self.cfg.all_arrived_bonus

        lidar_penalty = self._compute_lidar_proximity_penalty()
        inter_agent_penalty = self._compute_inter_agent_proximity_penalty()
        acceleration_penalty = self._compute_acceleration_penalty()
        step_term = np.full((self.num_agents,), self.cfg.step_penalty, dtype=np.float32)

        failure_term = np.zeros((self.num_agents,), dtype=np.float32)
        if failure:
            failure_term[:] = self.cfg.failure_penalty

        rewards = (
            progress_term
            + team_distance_term
            + arrival_term
            + lidar_penalty
            + inter_agent_penalty
            + acceleration_penalty
            + step_term
            + failure_term
        )
        rewards = rewards * self.cfg.reward_scale

        self.last_reward_terms = {
            "progress": progress_term.astype(np.float32),
            "team_distance": team_distance_term.astype(np.float32),
            "arrival": arrival_term.astype(np.float32),
            "lidar_proximity": lidar_penalty.astype(np.float32),
            "inter_agent_proximity": inter_agent_penalty.astype(np.float32),
            "acceleration": acceleration_penalty.astype(np.float32),
            "step": step_term.astype(np.float32),
            "failure": failure_term.astype(np.float32),
            "total": rewards.astype(np.float32),
        }
        return rewards.astype(np.float32)

    def _compute_lidar_proximity_penalty(self) -> np.ndarray:
        """
        Penalize being too close to obstacles according to LiDAR distances.
        """
        lidar_distances = self._compute_lidar_observations() * self.cfg.lidar_range
        min_dist = np.min(lidar_distances, axis=1)

        penalty = np.zeros((self.num_agents,), dtype=np.float32)
        danger_mask = min_dist < self.cfg.lidar_danger_distance
        warning_mask = (min_dist >= self.cfg.lidar_danger_distance) & (min_dist < self.cfg.lidar_warning_distance)

        penalty[danger_mask] += self.cfg.lidar_danger_penalty
        penalty[warning_mask] += -self.cfg.lidar_warning_penalty_scale * (
            self.cfg.lidar_warning_distance - min_dist[warning_mask]
        )
        return penalty.astype(np.float32)

    def _compute_inter_agent_proximity_penalty(self) -> np.ndarray:
        """Softly penalize UAVs that are close to other UAVs."""
        penalty = np.zeros((self.num_agents,), dtype=np.float32)

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.cfg.inter_agent_warning_distance:
                    pair_penalty = -self.cfg.inter_agent_warning_penalty_scale * (
                        self.cfg.inter_agent_warning_distance - dist
                    )
                    penalty[i] += pair_penalty
                    penalty[j] += pair_penalty

        return penalty.astype(np.float32)

    def _compute_acceleration_penalty(self) -> np.ndarray:
        """Penalize abrupt velocity and angular-velocity changes."""
        penalty = (
            -self.cfg.acceleration_penalty_scale_v * np.abs(self.linear_accelerations)
            -self.cfg.acceleration_penalty_scale_omega * np.abs(self.angular_accelerations)
        )
        return penalty.astype(np.float32)

    # ---------------------------------------------------------------------
    # Geometry, termination, diagnostics
    # ---------------------------------------------------------------------
    def _compute_assigned_target_distances(self) -> np.ndarray:
        assigned_targets = self.target_positions[self.assignments]
        return np.linalg.norm(self.positions - assigned_targets, axis=1).astype(np.float32)

    def _compute_assigned_target_vectors(self) -> np.ndarray:
        assigned_targets = self.target_positions[self.assignments]
        return (assigned_targets - self.positions).astype(np.float32)

    def _check_boundary_violation(self) -> bool:
        half_size = self.cfg.world_size / 2.0
        return bool(np.any(np.abs(self.positions) > half_size))

    def _check_obstacle_collision(self) -> bool:
        if self.cfg.num_obstacles == 0:
            return False

        diff = self.positions[:, None, :] - self.obstacle_centers[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        collision_thresholds = self.cfg.uav_radius + self.obstacle_radii + self.cfg.obstacle_safety_margin
        return bool(np.any(distances <= collision_thresholds[None, :]))

    def _check_inter_agent_collision(self) -> bool:
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
            reasons.append("success_all_targets_arrived")
        if boundary_violation:
            reasons.append("boundary_violation")
        if obstacle_collision:
            reasons.append("obstacle_collision")
        if inter_agent_collision:
            reasons.append("inter_agent_collision")
        if timeout:
            reasons.append("timeout")
        return "+".join(reasons) if reasons else ""

    def get_global_state(self) -> np.ndarray:
        """Return a simple global state vector for the future centralized critic."""
        state_parts = [
            self.positions.reshape(-1) / (self.cfg.world_size / 2.0),
            np.cos(self.headings),
            np.sin(self.headings),
            self.linear_velocities / self.cfg.max_linear_velocity,
            self.angular_velocities / self.cfg.max_angular_velocity,
            self.linear_accelerations / max(self.cfg.max_linear_velocity / self.cfg.dt, 1e-6),
            self.angular_accelerations / max(self.cfg.max_angular_velocity / self.cfg.dt, 1e-6),
            self.target_positions.reshape(-1) / (self.cfg.world_size / 2.0),
            self.assignments.astype(np.float32) / max(1, self.num_targets - 1),
        ]
        return np.concatenate(state_parts).astype(np.float32)

    def _get_info(self, **extra_flags) -> Dict:
        info = {
            "step_count": self.step_count,
            "done": self.done,
            "termination_reason": self.termination_reason,
            "positions": self.positions.copy(),
            "headings": self.headings.copy(),
            "linear_velocities": self.linear_velocities.copy(),
            "angular_velocities": self.angular_velocities.copy(),
            "linear_accelerations": self.linear_accelerations.copy(),
            "angular_accelerations": self.angular_accelerations.copy(),
            "trajectory_lengths": self.trajectory_lengths.copy(),
            "target_positions": self.target_positions.copy(),
            "assignments": self.assignments.copy(),
            "assignment_cost_matrix": self.assignment_cost_matrix.copy(),
            "assignment_info": dict(self.assignment_info),
            "arrived": self.arrived.copy(),
            "target_arrived": self.target_arrived.copy(),
            "obstacle_centers": self.obstacle_centers.copy(),
            "obstacle_radii": self.obstacle_radii.copy(),
            "obs_dim": self.obs_dim,
            "reward_terms": {k: v.copy() for k, v in self.last_reward_terms.items()},
        }
        info.update(extra_flags)
        return info

    @staticmethod
    def _wrap_angle(angle: np.ndarray) -> np.ndarray:
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    def render_state(self) -> None:
        print(f"Step: {self.step_count}")
        print(f"Done: {self.done}, reason: {self.termination_reason}")
        print(f"Assignments: {self.assignments.tolist()}")
        print(f"UAV arrived: {self.arrived.tolist()}")
        print(f"Target arrived: {self.target_arrived.tolist()}")
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

    def render_matplotlib(self, ax=None, show_lidar: bool = False):
        """Optional quick visualization using matplotlib."""
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

        ax.scatter(self.target_positions[:, 0], self.target_positions[:, 1], marker="*", s=160, label="Targets")
        ax.scatter(self.positions[:, 0], self.positions[:, 1], marker="o", label="UAVs")

        dx = np.cos(self.headings) * 0.5
        dy = np.sin(self.headings) * 0.5
        ax.quiver(self.positions[:, 0], self.positions[:, 1], dx, dy, angles="xy", scale_units="xy", scale=1)

        for i in range(self.num_agents):
            target = self.target_positions[self.assignments[i]]
            ax.plot([self.positions[i, 0], target[0]], [self.positions[i, 1], target[1]], linestyle="--", linewidth=1)
            ax.text(self.positions[i, 0], self.positions[i, 1], f"U{i}")
            ax.text(target[0], target[1], f"T{self.assignments[i]}")

        if show_lidar:
            self._draw_lidar(ax)

        ax.legend(loc="upper left")
        ax.grid(True)
        return ax

    def _draw_lidar(self, ax) -> None:
        """Draw LiDAR rays for debugging."""
        num_rays = self.cfg.lidar_num_rays
        half_fov = self.cfg.lidar_fov / 2.0
        relative_angles = np.linspace(-half_fov, half_fov, num_rays, dtype=np.float32)
        lidar = self._compute_lidar_observations() * self.cfg.lidar_range

        for i in range(self.num_agents):
            origin = self.positions[i]
            ray_angles = self.headings[i] + relative_angles
            for r_id, angle in enumerate(ray_angles):
                end = origin + lidar[i, r_id] * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                ax.plot([origin[0], end[0]], [origin[1], end[1]], linewidth=0.3, alpha=0.3)


if __name__ == "__main__":
    cfg = UAVEnvConfig(
        num_agents=3,
        num_targets=3,
        num_obstacles=5,
        assigner_name="hungarian",
        lidar_num_rays=35,
        seed=42,
    )
    env = MultiUAV2DEnv(cfg)

    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Expected obs dim:", env.obs_dim)
    print("Global state shape:", env.get_global_state().shape)
    env.render_state()

    total_reward = np.zeros(cfg.num_agents, dtype=np.float32)
    for _ in range(10):
        actions = np.array(
            [
                [1.0, 0.1],
                [1.0, 0.0],
                [1.0, -0.1],
            ],
            dtype=np.float32,
        )
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards
        env.render_state()
        print("Obs shape:", obs.shape)
        print("Rewards:", rewards, "Dones:", dones, "Reason:", info["termination_reason"])
        print("Target arrived:", info["target_arrived"])
        print("Reward terms:")
        for key, value in info["reward_terms"].items():
            print(f"  {key}: {value}")
        if dones.all():
            break

    print("\nTotal reward over test rollout:", total_reward)

    # Uncomment for visualization test:
    # import matplotlib.pyplot as plt
    # env.render_matplotlib(show_lidar=True)
    # plt.show()
