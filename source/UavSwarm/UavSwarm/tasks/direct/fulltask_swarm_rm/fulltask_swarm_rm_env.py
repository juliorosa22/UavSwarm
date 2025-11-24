# ============================================================
# SwarmQuadEnv (Isaac Lab 2.2.1)
# Direct-style MARL: multiple Crazyflies per environment
# ============================================================
### TODO adjust this env for full task swarm combined with Reward Machines
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.math import subtract_frame_transforms

from .fulltask_swarm_rm_env_cfg import FullTaskUAVSwarmEnvCfg

from isaaclab.markers import SPHERE_MARKER_CFG  # isort: skip

#check out https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html for more details on the functions implemented in DirectMARLEnv workflow
"""
Main idea of how use the Direct workflow when designing a task
###----- Overview of the Environment class structure:   
    _setup_scene(self): creates the environment, terrain, robots, etc. also  defines the distance between the parallel environments. This step is more the simulation scene configuration 
    _get_observations(self): gets the observations for each agent as a dict whe the obs should be in key {'policy':obs} when using a critic with different obs use {'policy':obs_policy,'critic':obs_critic}
    _pre_physics_step(self, actions): Mainly process the actions from the policy,performing computation like clipping or scaling, tranforming them to torques/forces, etc
    _apply_action(self): Apply the forces/torques to the robots, here is where the physics interaction happens
    _get_rewards(self): Compute the rewards for each agent, return as a dict with keys as agent names and values as reward tensors
    _get_dones(self): Compute the termination conditions for each agent, return as two dicts (terminated, time_out) with keys as agent names and values as boolean tensors
    _reset_idx(self, env_ids): Reset the environments specified by env_ids, resetting robot states, sampling new goals, etc

###-----Domain Randomization:
    Its also possible to implement domain randomization using the EventTerm and EventTermCfg
    Once the configclass for the randomization terms have been set up, the class must be added to the base config class for the task and be assigned to the variable events.

    @configclass
    class MyTaskConfig:
    events: EventCfg = EventCfg()

"""


class FullTaskUAVSwarmEnv(DirectMARLEnv):
    """
    Direct-style MARL environment with N Crazyflies per env.
    - Actions: per-drone [thrust, mx, my, mz] ⇒ shape (num_agents, 4)
    - Observations: per-drone 12 dims ⇒ shape (num_agents, 12)
    - Rewards: per-drone terms + swarm cohesion/collision penalties
    """

    cfg: FullTaskUAVSwarmEnvCfg

    def __init__(self, cfg: FullTaskUAVSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        
        self.num_drones = cfg.num_agents        
        self.global_step=0
        self.curriculum_stage=1
        self._obstacles_built=False
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

        # Create N Crazyflies per environment with distinct prim paths and materials
        for i in range(self.num_drones):
            robot_cfg: ArticulationCfg = self.cfg.robot_template.replace(
                prim_path=f"/World/envs/env_.*/Robot_{i}"
            ).replace(
                spawn=self.cfg.robot_template.spawn.replace(
                    visual_material_path=f"/World/Looks/Crazyflie_{i}"  # Unique path per drone
                )
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
    # as in the link tutorial: For asymmetric policies, the dictionary should also include the key critic and the states buffer as the value.
    def _post_physics_step(self):
        """Update curriculum stage based on global step."""
        self.global_step += 1
        # Update curriculum stage if applicable
        self.update_curriculum_stage()

    def update_curriculum_stage(self):
        step = self.global_step
        c = self.cfg.curriculum
        if step <= c.stage1_end:
            self.curriculum_stage = 1   # Hovering (individual)
        elif step <= c.stage2_end:
            self.curriculum_stage = 2   # Point-to-point (individual)
        elif step <= c.stage3_end:
            self.curriculum_stage = 3   # Point-to-point + obstacles
        elif step <= c.stage4_end:
            self.curriculum_stage = 4   # Swarm navigation
        else:
            self.curriculum_stage = 5   # Swarm + obstacles

    def _get_observations(self) -> dict:
        """Get observations for each drone (12 dims per drone).
        
        Returns:
            Dictionary mapping agent names directly to observations.
            Format: {"robot_0": obs0, "robot_1": obs1, "robot_2": obs2}
        """
        obs_dict = {}
        #obs_list = []
        
        for j, rob in enumerate(self._robots):
            agent_name = f"robot_{j}"
            
            # Transform desired position to body frame
            desired_pos_b, _ = subtract_frame_transforms(
                rob.data.root_pos_w, 
                rob.data.root_quat_w, 
                self._desired_pos_w[:, j, :]
            )
            
            # Concatenate observation components
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
            #obs_list.append(obs_j)
        
        # Concatenate all observations for centralized critic (state)
        #state = torch.cat(obs_list, dim=-1)  # (num_envs, num_agents * 12)
        return obs_dict
        #return {"policy": obs_dict, "critic": state}
        
    def _get_states(self) -> torch.Tensor:
        """Get centralized state for MAPPO critic.
        
        Returns:
            Concatenated observations from all agents for centralized critic.
            Shape: (num_envs, num_agents * obs_dim) = (num_envs, 3 * 12) = (num_envs, 36)
        """
        state_list = []
        
        for j, rob in enumerate(self._robots):
            # Transform desired position to body frame
            desired_pos_b, _ = subtract_frame_transforms(
                rob.data.root_pos_w, 
                rob.data.root_quat_w, 
                self._desired_pos_w[:, j, :]
            )
            
            # Same observation components as individual observations
            obs_j = torch.cat(
                [
                    rob.data.root_lin_vel_b,       # (num_envs, 3)
                    rob.data.root_ang_vel_b,       # (num_envs, 3)
                    rob.data.projected_gravity_b,  # (num_envs, 3)
                    desired_pos_b,                 # (num_envs, 3)
                ],
                dim=-1,
            )  # -> (num_envs, 12)
            
            state_list.append(obs_j)
        
        # Concatenate all agent observations into global state
        state = torch.cat(state_list, dim=-1)  # (num_envs, num_agents * 12)
        
        return state

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
        """Reset environments with inverted V formation for hovering task."""
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

        # Get environment origins
        env_origins = self._terrain.env_origins[env_ids]
        num_reset_envs = len(env_ids)

            # Set targets positions based on curriculum stage
        if self.curriculum_stage == 1:
            self._set_hover_goals(env_ids, env_origins)
        elif self.curriculum_stage == 2:
            self._set_individual_p2p_goals(env_ids, env_origins)
        elif self.curriculum_stage == 3:
            self._set_individual_p2p_goals_with_obstacles(env_ids, env_origins)
        elif self.curriculum_stage == 4:
            self._set_swarm_navigation_goals(env_ids, env_origins)
        else:  # stage 5
            self._set_swarm_navigation_goals_with_obstacles(env_ids, env_origins)

        # Optionally: reposition obstacles when used
        if self.curriculum_stage in (3, 5):
            self._ensure_obstacles_built()
            self._randomize_obstacles(env_ids, env_origins)

        ##OLD logic from V formation implementation
        # # Sample random spawn heights for each environment
        # spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
        #     self.cfg.spawn_height_range[0], 
        #     self.cfg.spawn_height_range[1]
        # )

        # # Calculate inverted V formation positions for all drones in all resetting environments
        # formation_positions = self._get_inverted_v_formation(env_ids, env_origins, spawn_heights)  
        # # Shape: (num_reset_envs, num_drones, 3)

        # # Reset each robot with formation-based positions
        # for j, rob in enumerate(self._robots):
        #     # Reset robot state
        #     joint_pos = rob.data.default_joint_pos[env_ids]
        #     joint_vel = rob.data.default_joint_vel[env_ids]
        #     default_root_state = rob.data.default_root_state[env_ids].clone()
            
        #     # Set position from precomputed formation
        #     default_root_state[:, :3] = formation_positions[:, j, :]  # (num_reset_envs, 3)
            
        #     # Write to simulation
        #     rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        #     rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        #     rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
        #     # Set goal position for hovering task:
        #     # Goal = spawn position + vertical offset + small XY noise
        #     self._desired_pos_w[env_ids, j, :2] = formation_positions[:, j, :2] + torch.zeros(
        #         num_reset_envs, 2, device=self.device
        #     ).uniform_(-self.cfg.goal_xy_noise, self.cfg.goal_xy_noise)
            
        #     self._desired_pos_w[env_ids, j, 2] = formation_positions[:, j, 2] + self.cfg.hover_height_offset

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable/disable debug visualization (matching copy_quadenv.py)."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizers"):
                self.goal_pos_visualizers = []
                for i in range(self.num_drones):
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.05
                    # Set color to green (R, G, B)
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 1.0, 0.0)
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
    

#### Helper Methods for adjusting agents targets positions based on curriculum stage

    def _set_hover_goals(self, env_ids, env_origins):
        # All drones should hover around a local origin, separated in XY to avoid collisions.
        # Example: keep XY close to origin, fixed Z height
        goal_z = self.cfg.goal_height
        xy_radius = 0.5  # small radius, but non-zero to avoid collisions

        for j, rob in enumerate(self._robots):
            # Place each drone slightly apart in XY
            offset_x = (j - self.num_drones / 2.0) * 0.5
            offset_y = 0.0
            self._desired_pos_w[env_ids, j, 0] = env_origins[:, 0] + offset_x
            self._desired_pos_w[env_ids, j, 1] = env_origins[:, 1] + offset_y
            self._desired_pos_w[env_ids, j, 2] = goal_z

    def _set_individual_p2p_goals(self, env_ids, env_origins):
        # Each drone starts at some point and must move to a *different* target,
        # far enough in XY to avoid collisions.
        goal_z = self.cfg.goal_height

        for j, rob in enumerate(self._robots):
            # Example: sample random distant goal along +X direction
            dx = torch.rand(len(env_ids), device=self.device) * 2.0 + 2.0  # [2,4] meters ahead
            dy = (j - self.num_drones / 2.0) * 0.5
            self._desired_pos_w[env_ids, j, 0] = env_origins[:, 0] + dx
            self._desired_pos_w[env_ids, j, 1] = env_origins[:, 1] + dy
            self._desired_pos_w[env_ids, j, 2] = goal_z


    def _ensure_obstacles_built(self):
        if self._obstacles_built:
            return
        # Here we would create some prims like cubes on /World/envs/env_.*/Obstacle_k
        # using sim_utils (e.g., CubeCfg, MeshCfg, etc.)
        # For now, just mark as built.
        self._obstacles_built = True

    def _randomize_obstacles(self, env_ids, env_origins):
        # Later: set obstacle positions in front of drones so that
        # goals are behind obstacles (as you specified).
        pass


    def get_inverted_v_formation(self, env_ids: torch.Tensor, env_origins: torch.Tensor, spawn_heights: torch.Tensor) -> torch.Tensor:
        """Calculate inverted V formation positions for all drones in specified environments.
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of the environments being reset, shape (num_reset_envs, 3)
            spawn_heights: Height offset for each environment, shape (num_reset_envs,)
        
        Returns:
            Tensor of shape (num_reset_envs, num_drones, 3) with absolute world positions 
            for each drone in each environment.
        """
        num_reset_envs = len(env_ids)
        
        # Inverted V: apex at front (negative X), wings spread backward and outward
        v_angle_rad = torch.deg2rad(torch.tensor(self.cfg.formation_v_angle_deg, device=self.device))
        base_sep = self.cfg.formation_base_separation
        
        # Scale separation based on max_num_agents to ensure formation fits
        scale_factor = max(1.0, self.cfg.min_sep / base_sep)
        effective_sep = base_sep * scale_factor
        
        # Generate formation positions template for each drone
        formation_template = torch.zeros(self.num_drones, 3, device=self.device)
        
        if self.num_drones == 1:
            # Single drone: centered at origin
            formation_template[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        else:
            # Multiple drones: inverted V formation
            # Apex drone (index 0) at the front (negative X)
            apex_idx = 0
            formation_template[apex_idx, 0] = self.cfg.formation_apex_offset  # X position (front)
            formation_template[apex_idx, 1] = 0.0  # Y position (centered)
            
            # Distribute remaining drones on left and right wings
            remaining_drones = self.num_drones - 1
            left_wing_count = remaining_drones // 2
            right_wing_count = remaining_drones - left_wing_count
            
            # Left wing (negative Y)
            for i in range(left_wing_count):
                wing_idx = i + 1
                x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)  # Backward
                y_offset = -(i + 1) * effective_sep * torch.sin(v_angle_rad)  # Left
                formation_template[wing_idx, 0] = self.cfg.formation_apex_offset + x_offset
                formation_template[wing_idx, 1] = y_offset
            
            # Right wing (positive Y)
            for i in range(right_wing_count):
                wing_idx = left_wing_count + i + 1
                x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)  # Backward
                y_offset = (i + 1) * effective_sep * torch.sin(v_angle_rad)  # Right
                formation_template[wing_idx, 0] = self.cfg.formation_apex_offset + x_offset
                formation_template[wing_idx, 1] = y_offset
    
        # Verify minimum separation constraint
        if self.num_drones > 1:
            dists = torch.cdist(formation_template.unsqueeze(0), formation_template.unsqueeze(0)).squeeze(0)
            # Set diagonal to large value to ignore self-distances
            dists = dists + torch.eye(self.num_drones, device=self.device) * 1000.0
            min_dist = dists.min()
            
            # If constraint violated, scale up the formation
            if min_dist < self.cfg.min_sep:
                scale_up = self.cfg.min_sep / min_dist
                formation_template[:, :2] *= scale_up
    
        # Expand template to all resetting environments
        # formation_template: (num_drones, 3)
        # Result: (num_reset_envs, num_drones, 3)
        formation_positions = formation_template.unsqueeze(0).expand(num_reset_envs, -1, -1).clone()
        
        # Add environment origins (XY) to all drones in each environment
        # env_origins: (num_reset_envs, 3)
        # Broadcast: (num_reset_envs, 1, 2) + (num_reset_envs, num_drones, 2)
        formation_positions[:, :, :2] += env_origins[:, :2].unsqueeze(1)
        
        # Add spawn heights (Z) to all drones in each environment
        # spawn_heights: (num_reset_envs,)
        # Broadcast: (num_reset_envs, 1) + (num_reset_envs, num_drones)
        formation_positions[:, :, 2] += spawn_heights.unsqueeze(1)
        
        return formation_positions  # Shape: (num_reset_envs, num_drones, 3)