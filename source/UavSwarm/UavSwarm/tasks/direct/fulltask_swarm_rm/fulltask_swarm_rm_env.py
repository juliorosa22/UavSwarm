# ============================================================
# SwarmQuadEnv (Isaac Lab 2.2.1)
# Direct-style MARL: multiple Crazyflies per environment
# ============================================================
### TODO adjust this env for full task swarm combined with Reward Machines
from __future__ import annotations

import torch
import isaacsim.core.utils.prims as prim_utils

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
        self.global_step=7_999_990
        self.curriculum_stage=4
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

        self.num_waypoints_per_agent = 3  # 3 obstacles → 3 waypoints (one behind each)
        
        #-----STAGE 3 Waypoint buffers
        # Waypoint paths: (num_envs, num_drones, num_waypoints, 3)
        self._waypoint_paths = torch.zeros(
            self.num_envs, self.num_drones, self.num_waypoints_per_agent, 3, 
            device=self.device
        )
        
        # Current waypoint index for each agent: (num_envs, num_drones)
        self._current_waypoint_idx = torch.zeros(
            self.num_envs, self.num_drones, dtype=torch.long, device=self.device
        )

        # -----------------------------------------------------
        # STAGE 5: Swarm waypoint buffers (shared across swarm)
        # -----------------------------------------------------
        self.num_swarm_waypoints = 3  # 3 waypoints through stacked X pattern
        
        # Swarm waypoint paths: (num_envs, num_waypoints, 3)
        # These are shared goals for the entire swarm (centroid targets)
        self._swarm_waypoint_paths = torch.zeros(
            self.num_envs, self.num_swarm_waypoints, 3,
            device=self.device
        )
        
        # Current swarm waypoint index: (num_envs,)
        self._current_swarm_waypoint_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Waypoint reached threshold (distance in meters)
        self.waypoint_reach_threshold = 0.3
        self.swarm_waypoint_reach_threshold = 0.5  # Slightly larger for swarm centroid

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

        # Build all obstacles before simulation
        self._build_all_obstacles()

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
        self.global_step += 1
        self.update_curriculum_stage()
        # Update waypoint goals for stage 3
        if self.curriculum_stage == 3:
            self._update_waypoint_goals()
        elif self.curriculum_stage == 5:
            self._update_swarm_waypoint_goals()

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


    ##TODO : adjust rewards weights and terms based on current stage in the curriculum
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


    ## TODO : adjust termination conditions based on current stage in the curriculum
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
                    rob.data.root_pos_w[:, 2] > 15.0
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
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with curriculum-dependent goals & scene adjustments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # -----------------------------------------------------
        # 1. LOG EPISODIC METRICS (same from your code)
        # -----------------------------------------------------
        final_distances = []
        for j in range(self.num_drones):
            dist = torch.linalg.norm(
                self._desired_pos_w[env_ids, j, :]
                - self._robots[j].data.root_pos_w[env_ids],
                dim=1,
            )
            final_distances.append(dist)
        final_distance_to_goal = torch.stack(final_distances).mean()

        # reward breakdown logs
        log_dict = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            log_dict[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # termination logs
        log_dict["Episode_Termination/died"] = torch.count_nonzero(
            self._last_terminated[env_ids]
        ).item()
        log_dict["Episode_Termination/time_out"] = torch.count_nonzero(
            self._last_timed_out[env_ids]
        ).item()
        log_dict["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        self.extras["log"] = log_dict

        # reset termination state
        self._last_terminated[env_ids] = False
        self._last_timed_out[env_ids] = False

        # -----------------------------------------------------
        # 2. RESET ROBOTS TO DEFAULT STATE
        # -----------------------------------------------------
        for rob in self._robots:
            rob.reset(env_ids)

        super()._reset_idx(env_ids)

        # desync episode starts
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0

        # -----------------------------------------------------
        # 3. GET ENV ORIGINS
        # -----------------------------------------------------
        env_origins = self._terrain.env_origins[env_ids]

        # -----------------------------------------------------
        # 4. CURRICULUM LOGIC — SELECT THE RIGHT STAGE BEHAVIOR
        # -----------------------------------------------------
        stage = self.curriculum_stage
        #print(f"Global step: {self.global_step}, Curriculum stage: {stage}")
        if stage == 1:
            # Individual Hover
            self._set_stage1_positions(env_ids, env_origins)
        elif stage == 2:
            # Individual XY movement
            self._set_stage2_positions(env_ids, env_origins)
        elif stage == 3:
            # Individual Point-to-Point with Obstacles
            self._set_stage3_positions(env_ids, env_origins)        
        elif stage == 4:
            # Swarm navigation (no obstacles)            
            self._set_stage4_positions(env_ids, env_origins)
        else:  # stage == 5
            self._set_stage5_positions(env_ids, env_origins)
            
            



#### Curriculum methods for adjusting agents targets positions and environment obstacles based on curriculum stage
    def update_curriculum_stage(self):
        step = self.global_step
        c = self.cfg.curriculum
        if step <= c.stage1_end:
            if self.curriculum_stage != 1:
                print("Changed to curriculum stage 1: Individual Hover")
            self.curriculum_stage = 1   # Hovering (individual)
        elif step <= c.stage2_end:
            if self.curriculum_stage != 2:
                print("Changed to curriculum stage 2: Individual Point-to-Point")
            self.curriculum_stage = 2   # Point-to-point (individual)
        elif step <= c.stage3_end:
            if self.curriculum_stage != 3:
                print("Changed to curriculum stage 3: Individual Point-to-Point with Obstacles")
            self.curriculum_stage = 3   # Point-to-point + obstacles
        elif step <= c.stage4_end:
            if self.curriculum_stage != 4:
                print("Changed to curriculum stage 4: Swarm Navigation without Obstacles")
            self.curriculum_stage = 4   # Swarm navigation
        else:
            if self.curriculum_stage != 5:
                print("Changed to curriculum stage 5: Swarm Navigation with Obstacles")
            self.curriculum_stage = 5   # Swarm + obstacles


    def _update_waypoint_goals(self):
        """Update goal positions based on waypoint progress for stage 3.
        When an agent reaches its current waypoint (within threshold distance),
        advance to the next waypoint in the path.
        """
        for env_idx in range(self.num_envs):
            for agent_idx, rob in enumerate(self._robots):
                # Get current waypoint index for this agent
                current_wp_idx = self._current_waypoint_idx[env_idx, agent_idx].item()
                
                # Check if agent has completed all waypoints
                if current_wp_idx >= self.num_waypoints_per_agent:
                    continue  # Already at final waypoint
                
                # Get agent's current position
                agent_pos = rob.data.root_pos_w[env_idx]  # (3,)
                
                # Get current waypoint goal
                current_waypoint = self._waypoint_paths[env_idx, agent_idx, current_wp_idx]  # (3,)
                
                # Calculate distance to current waypoint
                distance_to_waypoint = torch.linalg.norm(agent_pos - current_waypoint).item()
                
                # Check if waypoint is reached
                if distance_to_waypoint < self.waypoint_reach_threshold:
                    # Advance to next waypoint
                    next_wp_idx = current_wp_idx + 1
                    
                    if next_wp_idx < self.num_waypoints_per_agent:
                        # Update to next waypoint
                        self._current_waypoint_idx[env_idx, agent_idx] = next_wp_idx
                        self._desired_pos_w[env_idx, agent_idx, :] = self._waypoint_paths[env_idx, agent_idx, next_wp_idx, :]
                        
                        print(f"[Stage 3] Env {env_idx}, Agent {agent_idx}: Reached waypoint {current_wp_idx}, advancing to {next_wp_idx}")
                    else:
                        # Mark as completed (stay at last waypoint)
                        self._current_waypoint_idx[env_idx, agent_idx] = self.num_waypoints_per_agent
                        print(f"[Stage 3] Env {env_idx}, Agent {agent_idx}: Completed all waypoints!")
    
    def _update_swarm_waypoint_goals(self):
        """Update swarm goals based on centroid progress through waypoint path (Stage 5).
        
        When the swarm centroid reaches a waypoint, advance all agents to the next waypoint
        while maintaining their formation relative to the new target.
        """
        for env_idx in range(self.num_envs):
            # Get current waypoint index for this environment
            current_wp_idx = self._current_swarm_waypoint_idx[env_idx].item()
            
            # Check if swarm has completed all waypoints
            if current_wp_idx >= self.num_swarm_waypoints:
                continue  # Already at final waypoint
            
            # Calculate swarm centroid position
            swarm_positions = torch.stack([rob.data.root_pos_w[env_idx] for rob in self._robots], dim=0)  # (num_drones, 3)
            swarm_centroid = swarm_positions.mean(dim=0)  # (3,)
            
            # Get current waypoint target
            current_waypoint = self._swarm_waypoint_paths[env_idx, current_wp_idx]  # (3,)
            
            # Calculate distance from centroid to waypoint
            distance_to_waypoint = torch.linalg.norm(swarm_centroid - current_waypoint).item()
            
            # Check if waypoint is reached
            if distance_to_waypoint < self.swarm_waypoint_reach_threshold:
                # Advance to next waypoint
                next_wp_idx = current_wp_idx + 1
                
                if next_wp_idx < self.num_swarm_waypoints:
                    # Update to next waypoint
                    self._current_swarm_waypoint_idx[env_idx] = next_wp_idx
                    next_waypoint = self._swarm_waypoint_paths[env_idx, next_wp_idx]  # (3,)
                    
                    # Calculate formation offsets relative to current centroid
                    formation_offsets = swarm_positions - swarm_centroid.unsqueeze(0)  # (num_drones, 3)
                    
                    # Update each agent's goal to maintain formation around new waypoint
                    for j in range(self.num_drones):
                        goal_pos = next_waypoint + formation_offsets[j]
                        
                        self._desired_pos_w[env_idx, j, 0] = goal_pos[0]
                        self._desired_pos_w[env_idx, j, 1] = goal_pos[1]
                        self._desired_pos_w[env_idx, j, 2] = goal_pos[2]
                    
                    print(f"[Stage 5] Env {env_idx}: Swarm reached waypoint {current_wp_idx}, advancing to {next_wp_idx}")
                else:
                    # Mark as completed
                    self._current_swarm_waypoint_idx[env_idx] = self.num_swarm_waypoints
                    print(f"[Stage 5] Env {env_idx}: Swarm completed all waypoints!")

    ###---- Stage 1: Individual Hover ----###
    def _set_stage1_positions(self, env_ids, env_origins):
        """Set hover goals - simplified grid version."""
        ##This function places the drones in a grid formation at the start of the episode and assigns each drone a goal position directly above its start position at a certain height.
        num_reset_envs = len(env_ids)
        
        # Create grid of start positions
        grid_size = int(torch.ceil(torch.sqrt(torch.tensor(self.num_drones, dtype=torch.float32))))
        spacing = torch.zeros(1,device=self.device).uniform_(self.cfg.curriculum.spawn_grid_spacing_range[0], self.cfg.curriculum.spawn_grid_spacing_range[1])#max(0.5, self.cfg.min_sep)
        
        for env_idx in range(num_reset_envs):
            # Random permutation for this environment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            # Sample heights
            start_heights = torch.zeros(self.num_drones, device=self.device).uniform_(0.3, 0.6)
            min_height = self.cfg.curriculum.stage1_goal_height_range[0]
            max_height = self.cfg.curriculum.stage1_goal_height_range[1]
            goal_heights = torch.zeros(self.num_drones, device=self.device).uniform_(min_height, max_height) 
            
            for j, rob in enumerate(self._robots):
                env_id_single = env_ids[env_idx].unsqueeze(0)
                
                # Grid position
                grid_idx = perm[j].item()
                grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
                grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)
                
                # Reset robot
                joint_pos = rob.data.default_joint_pos[env_id_single]
                joint_vel = rob.data.default_joint_vel[env_id_single]
                default_root_state = rob.data.default_root_state[env_id_single].clone()
                
                # Start position
                default_root_state[:, 0] = env_origins[env_idx, 0] + grid_x
                default_root_state[:, 1] = env_origins[env_idx, 1] + grid_y
                default_root_state[:, 2] = start_heights[j]
                
                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # Goal position (minimal XY drift)
                xy_noise = torch.zeros(2, device=self.device).uniform_(-0.05, 0.05)
                self._desired_pos_w[env_id_single, j, 0] = default_root_state[0, 0] + xy_noise[0]
                self._desired_pos_w[env_id_single, j, 1] = default_root_state[0, 1] + xy_noise[1]
                self._desired_pos_w[env_id_single, j, 2] = goal_heights[j]

    ###---- Stage 2: Individual Point-to-Point ----###
    def _set_stage2_positions(self, env_ids, env_origins):
        """Set individual point-to-point goals for curriculum stage 2.
        
        Agents must learn to:
        1. Navigate stably in XY plane at different heights
        2. Rotate (yaw) to face the goal direction
        3. Reach distant goals while avoiding inter-agent collisions
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of environments, shape (num_reset_envs, 3)
        """
        num_reset_envs = len(env_ids)
        
        # Create grid of start positions
        grid_size = int(torch.ceil(torch.sqrt(torch.tensor(self.num_drones, dtype=torch.float32))))
        spacing = torch.zeros(1,device=self.device).uniform_(self.cfg.curriculum.spawn_grid_spacing_range[0], self.cfg.curriculum.spawn_grid_spacing_range[1])#max(0.5, self.cfg.min_sep)
        
        # Calculate height stratification to prevent collisions
        # Each drone operates on a different Z-plane
        z_spacing = self.cfg.curriculum.stage2_zdist_xy_plane
        base_height = 0.5  # Minimum start height
        
        for env_idx in range(num_reset_envs):
            # Random permutation for grid assignment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            # Assign heights to create vertical separation
            # Heights increase with drone index to create layered formation
            heights = torch.arange(self.num_drones, device=self.device, dtype=torch.float32)
            heights = base_height + heights * z_spacing
            
            # Randomize height assignment (shuffle which drone gets which height layer)
            height_perm = torch.randperm(self.num_drones, device=self.device)
            assigned_heights = heights[height_perm]
            
            for j, rob in enumerate(self._robots):
                env_id_single = env_ids[env_idx].unsqueeze(0)
                
                # -----------------------------------------------------
                # 1. START POSITION: Grid formation with unique heights
                # -----------------------------------------------------
                grid_idx = perm[j].item()
                grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
                grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)
                
                # Reset robot
                joint_pos = rob.data.default_joint_pos[env_id_single]
                joint_vel = rob.data.default_joint_vel[env_id_single]
                default_root_state = rob.data.default_root_state[env_id_single].clone()
                
                # Set start position
                start_x = env_origins[env_idx, 0] + grid_x
                start_y = env_origins[env_idx, 1] + grid_y
                start_z = assigned_heights[j]
                
                default_root_state[:, 0] = start_x
                default_root_state[:, 1] = start_y
                default_root_state[:, 2] = start_z
                
                # Write to simulation
                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # -----------------------------------------------------
                # 2. GOAL POSITION: Distant in XY, similar Z
                # -----------------------------------------------------
                # Sample distance and angle for goal
                goal_distance = torch.zeros(1, device=self.device).uniform_(
                    2.0, 
                    self.cfg.curriculum.stage2_goal_distance
                ).item()
                
                # Random angle in [0, 2π] to encourage yaw rotation
                # This ensures the drone must turn to face the goal
                goal_angle = torch.zeros(1, device=self.device).uniform_(
                    0.0, 
                    2.0 * torch.pi
                ).item()
                
                # Calculate goal XY offset from start position
                goal_offset_x = goal_distance * torch.cos(torch.tensor(goal_angle, device=self.device))
                goal_offset_y = goal_distance * torch.sin(torch.tensor(goal_angle, device=self.device))
                
                # Goal position: start + offset
                goal_x = start_x + goal_offset_x
                goal_y = start_y + goal_offset_y
                
                # Goal height: same as start with small noise
                goal_z_noise = torch.zeros(1, device=self.device).uniform_(-0.1, 0.1).item()
                goal_z = start_z + goal_z_noise
                
                # Set goal
                self._desired_pos_w[env_id_single, j, 0] = goal_x
                self._desired_pos_w[env_id_single, j, 1] = goal_y
                self._desired_pos_w[env_id_single, j, 2] = goal_z

    ###---- Stage 3: Individual Point-to-Point with Obstacles ----###
    def _set_stage3_positions(self, env_ids, env_origins):
        """Set individual obstacle course navigation with waypoint-based goals.
        
        Each agent navigates through a zig-zag obstacle course:
        - Waypoint 1: Behind first obstacle (center)
        - Waypoint 2: Behind second obstacle (left)
        - Waypoint 3: Behind third obstacle (right)
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of environments, shape (num_reset_envs, 3)
        """
        num_reset_envs = len(env_ids)
        offset_x, offset_y = self._get_stage_offset()  # Returns (20.0, 0.0) for stage 3
        
        # Obstacle course parameters (must match _build_stage3_obstacles)
        lane_width = 1.5
        obstacle_spacing_x = 3.0
        lateral_offset = 0.4
        course_start_x = offset_x + 3.0
        waypoint_distance_behind_obstacle = 1.0  # Distance behind each obstacle
        
        # Agent spawn parameters
        spawn_x = offset_x
        base_height = 0.8
        
        for env_idx in range(num_reset_envs):
            env_id_int = env_ids[env_idx].item()
            env_id_single = env_ids[env_idx].unsqueeze(0)
            
            # Random permutation for lane assignment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            for j, rob in enumerate(self._robots):
                # -----------------------------------------------------
                # AGENT START POSITION: At course entrance
                # -----------------------------------------------------
                agent_lane = perm[j].item()
                lane_center_y = offset_y + (agent_lane - (self.num_drones - 1) / 2.0) * lane_width
                
                start_x = env_origins[env_idx, 0] + spawn_x
                start_y = env_origins[env_idx, 1] + lane_center_y
                start_z = base_height
                
                # Reset robot
                joint_pos = rob.data.default_joint_pos[env_id_single]
                joint_vel = rob.data.default_joint_vel[env_id_single]
                default_root_state = rob.data.default_root_state[env_id_single].clone()
                
                default_root_state[:, 0] = start_x
                default_root_state[:, 1] = start_y
                default_root_state[:, 2] = start_z
                
                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # -----------------------------------------------------
                # GENERATE WAYPOINT PATH (3 waypoints)
                # -----------------------------------------------------
                for wp_idx in range(3):
                    # Obstacle position
                    obs_x = env_origins[env_idx, 0] + course_start_x + wp_idx * obstacle_spacing_x
                    
                    # Zig-zag pattern
                    if wp_idx == 0:
                        obs_y = env_origins[env_idx, 1] + lane_center_y  # Center
                    elif wp_idx == 1:
                        obs_y = env_origins[env_idx, 1] + lane_center_y - lateral_offset  # Left
                    else:  # wp_idx == 2
                        obs_y = env_origins[env_idx, 1] + lane_center_y + lateral_offset  # Right
                    
                    # Waypoint is behind obstacle
                    waypoint_x = obs_x + waypoint_distance_behind_obstacle
                    waypoint_y = obs_y
                    waypoint_z = start_z + torch.zeros(1, device=self.device).uniform_(-0.1, 0.1).item()
                    
                    # Store waypoint in path buffer
                    self._waypoint_paths[env_id_int, j, wp_idx, 0] = waypoint_x
                    self._waypoint_paths[env_id_int, j, wp_idx, 1] = waypoint_y
                    self._waypoint_paths[env_id_int, j, wp_idx, 2] = waypoint_z
                
                # Reset current waypoint index to 0 (first waypoint)
                self._current_waypoint_idx[env_id_int, j] = 0
                
                # Set initial goal to first waypoint
                self._desired_pos_w[env_id_single, j, :] = self._waypoint_paths[env_id_int, j, 0, :]
        
        print(f"[Stage 3] Initialized obstacle course with {self.num_waypoints_per_agent} waypoints per agent")    
   
    ##-- Build all the obstacle prims and its called in _setup_scene --##
    def _build_all_obstacles(self):
        """Build all obstacle prims at fixed locations for stages 3 and 5."""
        self._build_stage3_obstacles()
        self._build_stage5_obstacles()
        self._obstacles_built = True
        
    def _build_stage3_obstacles(self):
        """Build stage 3 obstacle course: 3 obstacles per agent in zig-zag pattern.
        
        Each agent gets 3 obstacles arranged in a zig-zag pattern:
        - Obstacle 1: Centered in agent's lane
        - Obstacle 2: Offset to left
        - Obstacle 3: Offset to right
        
        This forces agents to learn lateral maneuvering (zig-zag movement).
        """
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import (
            RigidBodyPropertiesCfg,
            CollisionPropertiesCfg,
        )
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        stage3_offset = (self.cfg.curriculum.stage3_offset_x, self.cfg.curriculum.stage3_offset_y, 0.0)
        source_env_idx = 0
        
        # Obstacle course parameters
        obstacle_size = (0.2, 0.8, 2.5)  # (thickness, width, height)
        base_height = 1.25  # Half of obstacle height
        
        # Lane configuration
        lane_width = 1.5  # Width of each agent's lane
        obstacle_spacing_x = 3.0  # Distance between obstacles along X-axis
        lateral_offset = 0.4  # How far obstacles shift left/right
        
        # Starting X position for obstacle course
        course_start_x = stage3_offset[0] + 3.0
        
        # Build obstacles for each agent
        for agent_idx in range(self.num_drones):
            # Calculate agent's lane center (Y coordinate)
            lane_center_y = stage3_offset[1] + (agent_idx - (self.num_drones - 1) / 2.0) * lane_width
            
            # Create 3 obstacles in zig-zag pattern for this agent
            for obs_idx in range(3):
                # Calculate obstacle position
                obs_x = course_start_x + obs_idx * obstacle_spacing_x
                
                # Zig-zag pattern: center, left, right
                if obs_idx == 0:
                    obs_y = lane_center_y  # Center
                elif obs_idx == 1:
                    obs_y = lane_center_y - lateral_offset  # Left
                else:  # obs_idx == 2
                    obs_y = lane_center_y + lateral_offset  # Right
                
                obs_z = base_height
                
                # Create obstacle prim
                wall_path = f"/World/envs/env_{source_env_idx}/Stage3_Agent{agent_idx}_Obs{obs_idx}"
                wall_cfg = CuboidCfg(
                    size=obstacle_size,
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.1, 0.1),
                        roughness=0.4,
                        metallic=0.0,
                    ),
                )
                wall_cfg.func(wall_path, wall_cfg, translation=(obs_x, obs_y, obs_z))
        
        print(f"[INFO] Built Stage 3 obstacle course:")
        print(f"  - {self.num_drones} agents × 3 obstacles = {self.num_drones * 3} total obstacles")
        print(f"  - Zig-zag pattern with {obstacle_spacing_x}m spacing")

    def _build_stage5_obstacles(self):
        """Build stage 5 obstacles in stacked X pattern.
        
        Creates 8 wall segments arranged in two X shapes stacked vertically:
        - Bottom X: 3 walls (left, center, right)
        - Top X: 5 walls (full X pattern)
        
        This forces swarm to navigate through strategic gaps while maintaining formation.
        """
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import (
            RigidBodyPropertiesCfg,
            CollisionPropertiesCfg,
        )
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        stage5_offset = (self.cfg.curriculum.stage5_offset_x, self.cfg.curriculum.stage5_offset_y, 0.0)
        source_env_idx = 0
        
        # Get offset configuration
        x_offset = self.cfg.curriculum.stage5_obsx_offset  # 2.0m
        y_offset = self.cfg.curriculum.stage5_obsy_offset  # 3.0m
        
        # Obstacle size: vertical walls blocking Y-axis travel
        obstacle_size = (1.2, 0.2, 2.5)  # (length, thickness, height)
        base_height = 1.25
        dist_from_spawn_swarm = 0  # Distance from swarm spawn area to obstacle pattern
        # Base Y position for the pattern
        base_y = stage5_offset[1]+dist_from_spawn_swarm
        base_x = stage5_offset[0]
        
        # Calculate wall positions based on stacked X pattern
        # Pattern layout (Y increases upward):
        #
        #     wall7        wall8         (Y = base + 2*y_offset)
        #           wall6               (Y = base + 1.5*y_offset) - center top X
        #     wall4        wall5         (Y = base + y_offset)
        #           wall3               (Y = base + 0.5*y_offset) - center bottom X
        #     wall1  wall2               (Y = base)
        
        wall_positions = [
            # Bottom X base (3 walls)
            (base_x - x_offset, base_y, base_height),              # wall1 - left bottom
            (base_x + x_offset, base_y, base_height),              # wall2 - right bottom
            (base_x, base_y + 0.5 * y_offset, base_height),        # wall3 - center bottom
            
            # Middle layer (2 walls)
            (base_x - x_offset, base_y + y_offset, base_height),   # wall4 - left middle
            (base_x + x_offset, base_y + y_offset, base_height),   # wall5 - right middle
            
            # Top X (3 walls)
            (base_x, base_y + 1.5 * y_offset, base_height),        # wall6 - center top
            (base_x - x_offset, base_y + 2 * y_offset, base_height), # wall7 - left top
            (base_x + x_offset, base_y + 2 * y_offset, base_height), # wall8 - right top
        ]
        
        # Build all 8 wall segments
        for wall_idx, (wall_x, wall_y, wall_z) in enumerate(wall_positions):
            wall_path = f"/World/envs/env_{source_env_idx}/WallSegment_Stage5_{wall_idx}"
            wall_cfg = CuboidCfg(
                size=obstacle_size,
                rigid_props=RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.9),
                    roughness=0.4,
                    metallic=0.0,
                ),
            )
            wall_cfg.func(wall_path, wall_cfg, translation=(wall_x, wall_y, wall_z))
        
        print(f"[INFO] Built Stage 5 stacked X obstacle pattern:")
        print(f"  - 8 wall segments in X formation")
        print(f"  - X offset: {x_offset}m, Y offset: {y_offset}m")
        print(f"  - Base position: ({base_x}, {base_y})")
    
    ###---- Stage 4: Swarm Navigation ----###
    #TODO adjust this stage to also use the swarm centroid waypoint logic
    def _set_stage4_positions(self, env_ids, env_origins):
        """Set swarm navigation goals with formation aligned to travel direction."""
        num_reset_envs = len(env_ids)
        
        spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
            self.cfg.curriculum.spawn_height_range[0], 
            self.cfg.curriculum.spawn_height_range[1]
        )
        
        # Start positions: Inverted V formation
        formation_positions = self.get_inverted_v_formation(env_ids, env_origins, spawn_heights)
        
        # Reset robots
        for j, rob in enumerate(self._robots):
            joint_pos = rob.data.default_joint_pos[env_ids]
            joint_vel = rob.data.default_joint_vel[env_ids]
            default_root_state = rob.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = formation_positions[:, j, :]
            
            rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Goal positions
        for env_idx in range(num_reset_envs):
            env_id_single = env_ids[env_idx]
            
            # Sample translation
            translation_distance = torch.zeros(1, device=self.device).uniform_(
                self.cfg.curriculum.swarm_translation_distance_min,
                self.cfg.curriculum.swarm_translation_distance_max
            ).item()
            
            translation_angle = torch.zeros(1, device=self.device).uniform_(0.0, 2.0 * torch.pi).item()
            
            translation_x = translation_distance * torch.cos(torch.tensor(translation_angle, device=self.device))
            translation_y = translation_distance * torch.sin(torch.tensor(translation_angle, device=self.device))
            
            # ALIGN V-formation to point toward travel direction
            # rotation_angle = translation_angle (V apex points toward goal)
            rotation_angle = translation_angle
            
            cos_theta = torch.cos(torch.tensor(rotation_angle, device=self.device))
            sin_theta = torch.sin(torch.tensor(rotation_angle, device=self.device))
            
            formation_center = formation_positions[env_idx].mean(dim=0)
            
            for j in range(self.num_drones):
                start_pos = formation_positions[env_idx, j]
                relative_pos = start_pos[:2] - formation_center[:2]
                
                # Rotate relative position
                rotated_x = cos_theta * relative_pos[0] - sin_theta * relative_pos[1]
                rotated_y = sin_theta * relative_pos[0] + cos_theta * relative_pos[1]
                
                # Translate to goal
                goal_x = formation_center[0] + translation_x + rotated_x
                goal_y = formation_center[1] + translation_y + rotated_y
                goal_z = start_pos[2] + torch.zeros(1, device=self.device).uniform_(-0.1, 0.1).item()
                
                self._desired_pos_w[env_id_single, j, 0] = goal_x
                self._desired_pos_w[env_id_single, j, 1] = goal_y
                self._desired_pos_w[env_id_single, j, 2] = goal_z

    ###---- Stage 5: Swarm Navigation with Obstacles ----###
    
    def _set_stage5_positions(self, env_ids, env_origins):
        """Set swarm waypoint navigation through stacked X obstacle pattern.
        
        Swarm navigates through 3 waypoints placed at the centers of the X pattern gaps:
        - Waypoint 1: Center of bottom X (between wall1, wall2, wall3)
        - Waypoint 2: Center of middle gap (between wall4, wall5, wall6)
        - Waypoint 3: Center of top X (between wall6, wall7, wall8)
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of environments, shape (num_reset_envs, 3)
        """
        num_reset_envs = len(env_ids)
        offset_x, offset_y = self._get_stage_offset()  # Returns (0.0, 20.0) for stage 5
        
        # Get obstacle configuration
        x_offset = self.cfg.curriculum.stage5_obsx_offset
        y_offset = self.cfg.curriculum.stage5_obsy_offset
        
        # Sample spawn heights
        spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
            self.cfg.curriculum.spawn_height_range[0], 
            self.cfg.curriculum.spawn_height_range[1]
        )
        
        # -----------------------------------------------------
        # 1. START POSITIONS: Inverted V formation at stage offset
        # -----------------------------------------------------
        offset_origins = env_origins.clone()
        offset_origins[:, 0] += offset_x
        offset_origins[:, 1] += offset_y
        
        formation_positions = self.get_inverted_v_formation(env_ids, offset_origins, spawn_heights)
        
        # Reset robots with formation positions
        for j, rob in enumerate(self._robots):
            joint_pos = rob.data.default_joint_pos[env_ids]
            joint_vel = rob.data.default_joint_vel[env_ids]
            default_root_state = rob.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = formation_positions[:, j, :]
            
            rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # -----------------------------------------------------
        # 2. SWARM WAYPOINT PATHS (3 waypoints through X pattern)
        # -----------------------------------------------------
        for env_idx in range(num_reset_envs):
            env_id_int = env_ids[env_idx].item()
            
            base_height = spawn_heights[env_idx].item()
            
            # Waypoint 1: Gap in bottom X (between wall3 and wall4/5)
            wp1_x = env_origins[env_idx, 0] + offset_x
            wp1_y = env_origins[env_idx, 1] + offset_y + 0.75 * y_offset  # Between center and middle
            wp1_z = base_height + torch.zeros(1, device=self.device).uniform_(-0.1, 0.1).item()
            
            self._swarm_waypoint_paths[env_id_int, 0, 0] = wp1_x
            self._swarm_waypoint_paths[env_id_int, 0, 1] = wp1_y
            self._swarm_waypoint_paths[env_id_int, 0, 2] = wp1_z
            
            # Waypoint 2: Gap in middle (between wall4/5 and wall6)
            wp2_x = env_origins[env_idx, 0] + offset_x
            wp2_y = env_origins[env_idx, 1] + offset_y + 1.25 * y_offset  # Between middle and center top
            wp2_z = base_height + torch.zeros(1, device=self.device).uniform_(-0.1, 0.1).item()
            
            self._swarm_waypoint_paths[env_id_int, 1, 0] = wp2_x
            self._swarm_waypoint_paths[env_id_int, 1, 1] = wp2_y
            self._swarm_waypoint_paths[env_id_int, 1, 2] = wp2_z
            
            # Waypoint 3: Final position beyond top X
            wp3_x = env_origins[env_idx, 0] + offset_x
            wp3_y = env_origins[env_idx, 1] + offset_y + 2.5 * y_offset  # Beyond wall7/8
            wp3_z = base_height + torch.zeros(1, device=self.device).uniform_(-0.1, 0.1).item()
            
            self._swarm_waypoint_paths[env_id_int, 2, 0] = wp3_x
            self._swarm_waypoint_paths[env_id_int, 2, 1] = wp3_y
            self._swarm_waypoint_paths[env_id_int, 2, 2] = wp3_z
            
            # Reset current waypoint index
            self._current_swarm_waypoint_idx[env_id_int] = 0
        
        # -----------------------------------------------------
        # 3. SET INITIAL GOALS (formation around first waypoint)
        # -----------------------------------------------------
        for env_idx in range(num_reset_envs):
            env_id_single = env_ids[env_idx]
            env_id_int = env_id_single.item()
            
            # Get first waypoint (swarm centroid target)
            swarm_target = self._swarm_waypoint_paths[env_id_int, 0, :]  # (3,)
            
            # Calculate formation center
            formation_center = formation_positions[env_idx].mean(dim=0)  # (3,)
            
            # Set individual goals to maintain formation relative to swarm target
            for j in range(self.num_drones):
                start_pos = formation_positions[env_idx, j]
                relative_pos = start_pos - formation_center
                
                # Goal = swarm_target + formation_offset
                goal_pos = swarm_target + relative_pos
                
                self._desired_pos_w[env_id_single, j, 0] = goal_pos[0]
                self._desired_pos_w[env_id_single, j, 1] = goal_pos[1]
                self._desired_pos_w[env_id_single, j, 2] = goal_pos[2]
        
        print(f"[Stage 5] Initialized swarm obstacle navigation with {self.num_swarm_waypoints} waypoints")


###---- Formation Helper Method: Inverted V Formation ----###

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
    
    def _get_stage_offset(self) -> tuple[float, float]:
        """Get the XY offset for the current curriculum stage.
        
        Returns:
            Tuple of (offset_x, offset_y) for the current stage
        """
        stage = self.curriculum_stage
        
        if stage == 3:
            return (self.cfg.curriculum.stage3_offset_x, self.cfg.curriculum.stage3_offset_y)
        elif stage == 5:
            return (self.cfg.curriculum.stage5_offset_x, self.cfg.curriculum.stage5_offset_y)
        else:
            # Stages 1, 2, 4: use default origin (0, 0)
            return (0.0, 0.0)