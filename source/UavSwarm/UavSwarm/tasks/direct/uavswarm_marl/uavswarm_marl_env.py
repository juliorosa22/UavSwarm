# ============================================================
# SwarmQuadEnv (Isaac Lab 2.2.1)
# Direct-style MARL: multiple Crazyflies per environment
# ============================================================

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.math import subtract_frame_transforms

from .uavswarm_marl_env_cfg import UavswarmMarlEnvCfg

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class UavswarmMarlEnv(DirectMARLEnv):
    """
    Direct-style MARL environment with N Crazyflies per env.
    - Actions: per-drone [thrust, mx, my, mz] ⇒ shape (num_agents, 4)
    - Observations: per-drone 12 dims ⇒ shape (num_agents, 12)
    - Rewards: per-drone terms + swarm cohesion/collision penalties
    """

    cfg: UavswarmMarlEnvCfg

    def __init__(self, cfg: UavswarmMarlEnvCfg, render_mode: str | None = None, **kwargs):
        
        self.num_drones = cfg.num_agents        
        
        # Initialize lists (before parent __init__)
        self._robots = []
        self._body_ids = []
        
        # Call parent constructor (triggers _setup_scene)
        super().__init__(cfg, render_mode, **kwargs)

        # Now device is available, initialize tensors
        self._actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, self.num_drones, 3, device=self.device)

        # Track termination reasons for logging
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "formation", "collision"]
        }

        # Get body indices and masses after robots are created
        self._body_ids = [rob.find_bodies("body")[0] for rob in self._robots]
        masses = [rob.root_physx_view.get_masses()[0].sum() for rob in self._robots]
        self._masses = torch.tensor(masses, device=self.device).view(1, self.num_drones)
        
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weights = (self._masses * self._gravity_magnitude).squeeze(0)  # (num_drones,)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Setup the scene with terrain and N robots per environment."""
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create N Crazyflies per environment with distinct prim paths
        for i in range(self.num_drones):
            robot_cfg: ArticulationCfg = self.cfg.robot_template.replace(
                prim_path=f"/World/envs/env_.*/Robot_{i}"
            )
            robot = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """Convert actions to thrust and moments for each drone.
        
        Args:
            actions: Dictionary mapping agent names to action tensors.
                     Each tensor has shape (num_envs, action_dim)
        """
        # Convert dictionary to stacked tensor: (num_envs, num_drones, 4)
        actions_list = []
        for i in range(self.num_drones):
            agent_name = f"robot_{i}"
            actions_list.append(actions[agent_name])
        
        actions_tensor = torch.stack(actions_list, dim=1)  # (num_envs, num_drones, 4)
        self._actions = actions_tensor.clone().clamp(-1.0, 1.0)
        
        # Process each drone's actions
        for j in range(self.num_drones):
            thrust_cmd = (self._actions[:, j, 0] + 1.0) / 2.0  # [0, 1]
            self._thrust[:, j, 0, 2] = self.cfg.thrust_to_weight * self._robot_weights[j] * thrust_cmd
            self._moment[:, j, 0, :] = self.cfg.moment_scale * self._actions[:, j, 1:]

    def _apply_action(self):
        """Apply forces and torques to each robot."""
        for j, rob in enumerate(self._robots):
            rob.set_external_force_and_torque(
                self._thrust[:, j, :, :], 
                self._moment[:, j, :, :], 
                body_ids=self._body_ids[j]
            )

    def _get_observations(self) -> dict:
        """Get observations for each drone (12 dims per drone).
        
        Returns:
            Dictionary with observations in the format expected by DirectMARLEnv.
            For each agent, obs has shape (num_envs, obs_dim)
        """
        obs_dict = {}
        
        for j, rob in enumerate(self._robots):
            agent_name = f"robot_{j}"
            
            # Transform desired position to body frame
            desired_pos_b, _ = subtract_frame_transforms(
                rob.data.root_pos_w, 
                rob.data.root_quat_w, 
                self._desired_pos_w[:, j, :]
            )
            
            # Concatenate observation components (matching copy_quadenv.py)
            obs_j = torch.cat(
                [
                    rob.data.root_lin_vel_b,       # (num_envs, 3)
                    rob.data.root_ang_vel_b,       # (num_envs, 3)
                    rob.data.projected_gravity_b,  # (num_envs, 3)
                    desired_pos_b,                 # (num_envs, 3)
                ],
                dim=-1,
            )  # -> (num_envs, 12)
            
            obs_dict[agent_name] = obs_j
        
        return {"policy": obs_dict}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Calculate rewards matching copy_quadenv.py logic + swarm terms.
        
        Returns:
            Dictionary mapping agent names to reward tensors of shape (num_envs,)
        """
        lin_vel_sum = torch.zeros(self.num_envs, device=self.device)
        ang_vel_sum = torch.zeros(self.num_envs, device=self.device)
        dist_goal_sum = torch.zeros(self.num_envs, device=self.device)

        pos_list = []

        # Per-drone rewards (matching copy_quadenv.py)
        for j, rob in enumerate(self._robots):
            lin_vel = torch.sum(torch.square(rob.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(rob.data.root_ang_vel_b), dim=1)
            
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, j, :] - rob.data.root_pos_w, dim=1
            )
            distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

            lin_vel_sum += lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
            ang_vel_sum += ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
            dist_goal_sum += distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt

            pos_list.append(rob.data.root_pos_w)

        # Swarm-specific penalties
        positions = torch.stack(pos_list, dim=1)  # (num_envs, num_drones, 3)
        dmat = torch.cdist(positions, positions)  # pairwise distances

        # Formation penalty: too close or too dispersed
        too_close = torch.clamp(self.cfg.min_sep - dmat, min=0.0)
        mean_dist = torch.mean(dmat, dim=(1, 2))
        formation_pen = self.cfg.formation_penalty_scale * (
            torch.mean(too_close, dim=(1, 2)) + 
            torch.clamp(mean_dist - self.cfg.max_sep_mean, min=0.0)
        ) * self.step_dt

        # Collision penalty: any drone below minimum height
        collided = positions[:, :, 2] < 0.1
        collision_pen = self.cfg.collision_penalty_scale * collided.any(dim=1).float() * self.step_dt

        # Total reward (shared across all agents)
        reward = lin_vel_sum + ang_vel_sum + dist_goal_sum + formation_pen + collision_pen

        # Logging (matching copy_quadenv.py pattern)
        self._episode_sums["lin_vel"] += lin_vel_sum
        self._episode_sums["ang_vel"] += ang_vel_sum
        self._episode_sums["distance_to_goal"] += dist_goal_sum
        self._episode_sums["formation"] += formation_pen
        self._episode_sums["collision"] += collision_pen

        # Return dictionary with same reward for all agents (cooperative setting)
        return {f"robot_{i}": reward for i in range(self.num_drones)}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Check termination conditions (matching copy_quadenv.py).
        
        Returns:
            Tuple of (terminated_dict, time_out_dict) where each is a dictionary
            mapping agent names to boolean tensors of shape (num_envs,)
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Died if any drone is too low or too high
        died_list = []
        for rob in self._robots:
            died_list.append(
                torch.logical_or(
                    rob.data.root_pos_w[:, 2] < 0.1, 
                    rob.data.root_pos_w[:, 2] > 2.0
                )
            )
        died = torch.stack(died_list, dim=1).any(dim=1)
        
        # Store for logging in reset
        self._last_terminated = died
        self._last_timed_out = time_out
        
        # Return dictionaries for MARL (all agents share same termination conditions)
        terminated_dict = {f"robot_{i}": died for i in range(self.num_drones)}
        time_out_dict = {f"robot_{i}": time_out for i in range(self.num_drones)}
        
        return terminated_dict, time_out_dict

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments (matching copy_quadenv.py pattern)."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # Logging (matching copy_quadenv.py)
        final_distances = []
        for j in range(self.num_drones):
            dist = torch.linalg.norm(
                self._desired_pos_w[env_ids, j, :] - self._robots[j].data.root_pos_w[env_ids], 
                dim=1
            )
            final_distances.append(dist)
        final_distance_to_goal = torch.stack(final_distances).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = {}
        self.extras["log"].update(extras)
        
        extras = {}
        # Use stored termination info for logging
        extras["Episode_Termination/died"] = torch.count_nonzero(self._last_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self._last_timed_out[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # Reset termination tracking
        self._last_terminated[env_ids] = False
        self._last_timed_out[env_ids] = False

        # Reset all robots
        for rob in self._robots:
            rob.reset(env_ids)
        
        super()._reset_idx(env_ids)
        
        # Spread out resets (matching copy_quadenv.py)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0

        # Sample new goals and reset robot states
        env_origins = self._terrain.env_origins[env_ids]
        x_step, y_slope = self.cfg.v_spacing_xy

        for j, rob in enumerate(self._robots):
            # Sample goal positions (matching copy_quadenv.py)
            self._desired_pos_w[env_ids, j, :2] = torch.zeros_like(
                self._desired_pos_w[env_ids, j, :2]
            ).uniform_(-self.cfg.goal_xy_range, self.cfg.goal_xy_range)
            self._desired_pos_w[env_ids, j, :2] += env_origins[:, :2]
            self._desired_pos_w[env_ids, j, 2] = torch.zeros_like(
                self._desired_pos_w[env_ids, j, 2]
            ).uniform_(0.5, 1.5)

            # Reset robot state with V-formation
            joint_pos = rob.data.default_joint_pos[env_ids]
            joint_vel = rob.data.default_joint_vel[env_ids]
            default_root_state = rob.data.default_root_state[env_ids]
            default_root_state[:, :3] += env_origins
            
            # V-formation offset
            default_root_state[:, 0] += j * x_step
            default_root_state[:, 1] += (j - self.num_drones / 2.0) * y_slope
            
            rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable/disable debug visualization (matching copy_quadenv.py)."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizers"):
                self.goal_pos_visualizers = []
                for i in range(self.num_drones):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                    marker_cfg.prim_path = f"/Visuals/Command/goal_position_{i}"
                    self.goal_pos_visualizers.append(VisualizationMarkers(marker_cfg))
            
            for viz in self.goal_pos_visualizers:
                viz.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizers"):
                for viz in self.goal_pos_visualizers:
                    viz.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization markers (matching copy_quadenv.py)."""
        if hasattr(self, "goal_pos_visualizers"):
            for i, viz in enumerate(self.goal_pos_visualizers):
                viz.visualize(self._desired_pos_w[:, i, :])