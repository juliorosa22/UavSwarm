# ============================================================
# SwarmQuadEnv (Isaac Lab 2.3.0)
# Direct-style MARL: multiple Crazyflies per environment using Curriculum Learning 
# Authors: Julio Rosa, adapted from CopyQuadEnv by NVIDIA Isaac Sim Team
# ============================================================
### TODO adjust this env for full task swarm combined with Reward Machines
from __future__ import annotations

import torch
from isaaclab.utils.math import subtract_frame_transforms
from .fulltask_swarm_rm_env_cfg import FullTaskUAVSwarmEnvCfg
from isaaclab.markers import SPHERE_MARKER_CFG  # isort: skip
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv

#import isaacsim.core.utils.prims as prim_utils
#from isaaclab.markers import VisualizationMarkers
#from isaaclab.scene import InteractiveSceneCfg



#from isaaclab.sensors import RayCaster
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
    # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
    @configclass
    class MyTaskConfig:
    events: EventCfg = EventCfg()

"""

#TODO Remove all the usage of ray_caster. after some on NVIDIA Isaac Sim team confirmed that the ray caster sensor is not working properly with multiple obstacles yet and MARL envs.
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
        
        self._obstacles_built=False
        self.curriculum_stage = 1#cfg.curriculum.active_stage
        
        cfg.episode_length_s = cfg.curriculum.get_episode_length()
        print(f"[INFO] Stage {cfg.curriculum.active_stage}: Episode length = {cfg.episode_length_s}s")
        # Initialize lists (before parent __init__)


        self._robots = []
        self._body_ids = []
        self._obstacle_positions=None
        super().__init__(cfg, render_mode, **kwargs)
        
        # Now device is available, initialize tensors
        self._actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, self.num_drones, 3, device=self.device)

        # Track termination reasons for logging
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ✅ NEW: Reward Machine state buffers
        # State encoding: 0=Hovering(H), 1=Single-moving(S), 2=Coop-moving(C), 3=Obstacle-avoiding(O)
        self._rm_states = torch.zeros(self.num_envs, self.num_drones, dtype=torch.long, device=self.device)
        # State names for debugging/logging
        self._rm_state_names = ['H', 'S', 'C', 'O']


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
        self._swarm_centroid = torch.zeros(self.num_envs, 3, device=self.device)
        self.waypoint_reach_threshold = 0.3
        self.swarm_waypoint_reach_threshold = 0.5  # Slightly larger for swarm centroid

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    ## MAIN ENVIRONMENT FUNCTIONS ##

    def _setup_scene(self):
        """Setup the scene with terrain and N robots per environment."""
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create N Crazyflies per environment AND their sensors
        for i in range(self.num_drones):
            # Robot configuration
            robot_cfg: ArticulationCfg = self.cfg.robot_template.replace(
                prim_path=f"/World/envs/env_.*/Robot_{i}"
            ).replace(
                spawn=self.cfg.robot_template.spawn.replace(
                    visual_material_path=f"/World/Looks/Crazyflie_{i}"
                )
            )
            robot = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)
            
            #No more ray casters for now as they do not work properly with multiple obstacles and MARL envs
            
        # ✅ NEW: Build obstacles ONLY for active stage
        if self.curriculum_stage == 3:
            print("[INFO] Building Stage 3 obstacles at origin...")
            self._build_stage3_obstacles_at_origin()
        elif self.curriculum_stage == 5:
            print("[INFO] Building Stage 5 obstacles at origin...")
            self._build_stage5_obstacles_at_origin()
        else:
            print(f"[INFO] Stage {self.curriculum_stage} has no obstacles")
        

        # ✅ Clone environments (will replicate sensors automatically)
        self.scene.clone_environments(copy_from_source=False)
    
    
        # Filter collisions
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """Convert actions to thrust and moments for each drone.
        
        Args:
            actions: Dictionary mapping agent names to action tensors.
                     Each tensor has shape (num_envs, action_dim)
        """
        #self.global_step += 1
        #self.update_curriculum_stage()
        # Update waypoint goals for stage 3
        if self.curriculum_stage == 3:
            self._update_waypoint_goals()
        elif self.curriculum_stage == 5:
            self._update_swarm_waypoint_goals()

         # Compute swarm centroid for stages 4 and 5 (for visualization and waypoint logic)
        if self.curriculum_stage in [4, 5]:
            self._compute_swarm_centroid()

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
        # ✅ NEW: Reset RM states to Hovering (0) for all agents
        self._rm_states[env_ids, :] = 0
        # -----------------------------------------------------
        # 3. GET ENV ORIGINS
        # -----------------------------------------------------
        env_origins = self._terrain.env_origins[env_ids]

        # -----------------------------------------------------
        # 4. CURRICULUM LOGIC — SELECT THE RIGHT STAGE BEHAVIOR
        # -----------------------------------------------------
        # ✅ NEW: Only handle active stage (no offsets)
        stage = self.curriculum_stage
        
        if stage == 1:
            self._set_stage1_positions(env_ids, env_origins)
        elif stage == 2:
            self._set_stage2_positions(env_ids, env_origins)
        elif stage == 3:
            self._set_stage3_positions(env_ids, env_origins)
        elif stage == 4:
            self._set_stage4_positions(env_ids, env_origins)
        elif stage == 5:
            self._set_stage5_positions(env_ids, env_origins)

    def _get_observations(self) -> dict:
        """Fully vectorized observation generation."""
        self._switch_rm_state()
        
        # Stack all robot data: (num_drones, num_envs, 3)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_gravities = torch.stack([rob.data.projected_gravity_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)
        
        # Transform desired positions to body frame (vectorized for all agents)
        # _desired_pos_w: (num_envs, num_drones, 3) -> transpose -> (num_drones, num_envs, 3)
        desired_pos_w_transposed = self._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)
        
        # Vectorized frame transformation for all agents at once
        desired_pos_b, _ = subtract_frame_transforms(
            all_positions.reshape(-1, 3),  # (num_drones*num_envs, 3)
            all_quats.reshape(-1, 4),
            desired_pos_w_transposed.reshape(-1, 3)
        )
        desired_pos_b = desired_pos_b.reshape(self.num_drones, self.num_envs, 3)
        
        # Get obstacle distances (vectorized)
        obstacle_dists = self._get_nearest_obstacle_distance_vectorized(all_positions)  # (num_drones, num_envs)
        
        # Get neighbor data (vectorized)
        neighbor_rel_pos_b, neighbor_rel_vel_b = self._get_nearest_neighbor_data_vectorized(
            all_positions, all_quats
        )  # Each: (num_drones, num_envs, 3)
        
        # RM state one-hot encoding (vectorized)
        # _rm_states: (num_envs, num_drones) -> transpose -> (num_drones, num_envs)
        rm_states_transposed = self._rm_states.transpose(0, 1)  # (num_drones, num_envs)
        rm_state_onehot = torch.nn.functional.one_hot(
            rm_states_transposed, 
            num_classes=self.cfg.reward_cfg.num_rm_states
        ).float()  # (num_drones, num_envs, 4)
        
        # Concatenate all observations: (num_drones, num_envs, 23)
        all_obs = torch.cat([
            all_lin_vels,                           # (num_drones, num_envs, 3)
            all_ang_vels,                           # (num_drones, num_envs, 3)
            all_gravities,                          # (num_drones, num_envs, 3)
            desired_pos_b,                          # (num_drones, num_envs, 3)
            obstacle_dists.unsqueeze(-1),           # (num_drones, num_envs, 1)
            neighbor_rel_vel_b,                     # (num_drones, num_envs, 3)
            neighbor_rel_pos_b,                     # (num_drones, num_envs, 3)
            rm_state_onehot,                        # (num_drones, num_envs, 4)
        ], dim=-1)  # (num_drones, num_envs, 23)
        
        # Convert to dictionary format (still need loop, but much faster than per-agent computation)
        obs_dict = {}
        for j in range(self.num_drones):
            obs_dict[f"robot_{j}"] = all_obs[j]  # (num_envs, 23)
        
        return obs_dict

    def _get_states(self) -> torch.Tensor:
        """Get centralized state for MAPPO critic (fully vectorized).
        
        Returns:
            Concatenated observations from all agents for centralized critic.
            Shape: (num_envs, num_agents * obs_dim)
        """
        # ✅ REUSE observations already computed in _get_observations()
        # Since _get_observations() is called before _get_states() in the environment step,
        # we can reuse the vectorized computation
        
        # Stack all robot data: (num_drones, num_envs, 3)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_gravities = torch.stack([rob.data.projected_gravity_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)
        
        # Transform desired positions to body frame (vectorized for all agents)
        desired_pos_w_transposed = self._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)
        
        desired_pos_b, _ = subtract_frame_transforms(
            all_positions.reshape(-1, 3),
            all_quats.reshape(-1, 4),
            desired_pos_w_transposed.reshape(-1, 3)
        )
        desired_pos_b = desired_pos_b.reshape(self.num_drones, self.num_envs, 3)
        
        # Get obstacle distances (vectorized)
        obstacle_dists = self._get_nearest_obstacle_distance_vectorized(all_positions)
        
        # Get neighbor data (vectorized)
        neighbor_rel_pos_b, neighbor_rel_vel_b = self._get_nearest_neighbor_data_vectorized(
            all_positions, all_quats
        )
        
        # RM state one-hot encoding (vectorized)
        rm_states_transposed = self._rm_states.transpose(0, 1)
        rm_state_onehot = torch.nn.functional.one_hot(
            rm_states_transposed, 
            num_classes=self.cfg.reward_cfg.num_rm_states
        ).float()
        
        # Concatenate all observations: (num_drones, num_envs, 23)
        all_obs = torch.cat([
            all_lin_vels,
            all_ang_vels,
            all_gravities,
            desired_pos_b,
            obstacle_dists.unsqueeze(-1),
            neighbor_rel_vel_b,
            neighbor_rel_pos_b,
            rm_state_onehot,
        ], dim=-1)
        
        # ✅ VECTORIZED CONCATENATION: Reshape to (num_envs, num_agents * obs_dim)
        # Transpose to (num_envs, num_drones, 23)
        all_obs = all_obs.transpose(0, 1)
        
        # Flatten agents dimension: (num_envs, num_drones * 23)
        state = all_obs.reshape(self.num_envs, -1)
        
        return state

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Fully vectorized reward calculation."""
        
        # Stack all robot data: (num_drones, num_envs, 3)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        
        # Base velocity penalties (vectorized)
        lin_vel_sq = torch.sum(all_lin_vels ** 2, dim=2)  # (num_drones, num_envs)
        ang_vel_sq = torch.sum(all_ang_vels ** 2, dim=2)  # (num_drones, num_envs)
        
        # State-dependent scaling (vectorized)
        # _rm_states: (num_envs, num_drones) -> transpose -> (num_drones, num_envs)
        is_hovering = (self._rm_states.transpose(0, 1) == 0).float()  # (num_drones, num_envs)
        vel_scale = 1.0 + 0.5 * is_hovering
        
        # Apply penalties
        lin_vel_sum = (lin_vel_sq * vel_scale).sum(dim=0) * self.cfg.reward_cfg.lin_vel_reward_scale * self.step_dt
        ang_vel_sum = (ang_vel_sq * vel_scale).sum(dim=0) * self.cfg.reward_cfg.ang_vel_reward_scale * self.step_dt
        
        # Distance to goal (vectorized)
        # _desired_pos_w: (num_envs, num_drones, 3) -> transpose -> (num_drones, num_envs, 3)
        desired_transposed = self._desired_pos_w.transpose(0, 1)
        distances = torch.linalg.norm(desired_transposed - all_positions, dim=2)  # (num_drones, num_envs)
        distance_mapped = 1 - torch.tanh(distances / 0.8)
        dist_goal_sum = distance_mapped.sum(dim=0) * self.cfg.reward_cfg.distance_to_goal_reward_scale * self.step_dt
        
        # Formation penalties (already vectorized)
        if self.curriculum_stage in [4, 5]:
            # Transpose to (num_envs, num_drones, 3)
            positions = all_positions.transpose(0, 1)
            dmat = torch.cdist(positions, positions)
            too_close = torch.clamp(self.cfg.swarm_cfg.min_safe_distance - dmat, min=0.0)
            mean_dist = torch.mean(dmat, dim=(1, 2))
            formation_pen = self.cfg.reward_cfg.formation_penalty_scale * (
                torch.mean(too_close, dim=(1, 2)) + 
                torch.clamp(mean_dist - self.cfg.swarm_cfg.max_formation_distance, min=0.0)
            ) * self.step_dt
        else:
            formation_pen = torch.zeros(self.num_envs, device=self.device)
        
        # Collision penalty (vectorized)
        # all_positions: (num_drones, num_envs, 3) -> z-coords: (num_drones, num_envs)
        agent_z = all_positions[:, :, 2]
        collided = agent_z < 0.1  # (num_drones, num_envs)
        collision_pen = self.cfg.reward_cfg.collision_penalty_scale * collided.any(dim=0).float() * self.step_dt
        
        # RM state bonuses (vectorized)
        altitude_bonus = self._compute_altitude_bonus_vectorized(all_positions, desired_transposed)
        obstacle_bonus = self._compute_obstacle_bonus_vectorized(all_positions)
        neighbor_bonus = self._compute_neighbor_bonus_vectorized(all_positions)
        state_bonus = self._compute_state_bonus_vectorized()
        
        # Combine
        reward = (
            lin_vel_sum + ang_vel_sum + dist_goal_sum +
            formation_pen + collision_pen +
            altitude_bonus + obstacle_bonus + neighbor_bonus + state_bonus
        )
        
        # Logging (vectorized)
        self._episode_sums["lin_vel"] += lin_vel_sum
        self._episode_sums["ang_vel"] += ang_vel_sum
        self._episode_sums["distance_to_goal"] += dist_goal_sum
        self._episode_sums["formation"] += formation_pen
        self._episode_sums["collision"] += collision_pen
        
        return {f"robot_{i}": reward for i in range(self.num_drones)}


#------Disabled RayCaster related functions for now as the sensor is not stable yet with multiple obstacles and MARL envs.
    def _get_raycaster_data(self, robot_index: int) -> torch.Tensor:
        """Get RayCaster distance measurements for a specific robot.
        
        Args:
            robot_index: Index of the robot (0 to num_drones-1)
        
        Returns:
            Tensor of ray distances, shape (num_envs, num_rays)
            Invalid rays are clamped to max_distance
        """
        ray_caster = self._ray_casters[robot_index]
        
        # Check if data exists and has correct shape
        if hasattr(ray_caster.data, 'ray_distances') and ray_caster.data.ray_distances is not None:
            ray_distances = ray_caster.data.ray_distances  # Shape: (num_envs, num_rays)
            
            # Handle invalid rays (rays that didn't hit anything)
            # Invalid rays return -1.0, replace with max_distance
            ray_distances = torch.where(
                ray_distances < 0.0,
                torch.full_like(ray_distances, self.cfg.ray_max_distance),
                ray_distances
            )
            
            # Clamp to valid range [0, max_distance]
            ray_distances = torch.clamp(ray_distances, 0.0, self.cfg.ray_max_distance)
        else:
            # ✅ FALLBACK: If sensor not ready, return max distances
            ray_distances = torch.full(
                (self.num_envs, self.cfg.num_rays), 
                self.cfg.ray_max_distance, 
                device=self.device
            )
        
        return ray_distances

    def _build_single_observation(self, robot_idx: int) -> torch.Tensor:
        """Build observation vector for a single robot (used by both policy and critic).
        
        Args:
            robot_idx: Index of the robot (0 to num_drones-1)
        
        Returns:
            Observation tensor of shape (num_envs, 23):
            - [0:3]: Linear velocity (body frame)
            - [3:6]: Angular velocity (body frame)
            - [6:9]: Projected gravity (body frame)
            - [9:12]: Desired position (body frame)
            - [12]: Nearest obstacle distance
            - [13:16]: Neighbor relative velocity (body frame)
            - [16:19]: Neighbor relative position (body frame)
            - [19:23]: RM state (one-hot encoded)
        """
        rob = self._robots[robot_idx]
        
        # Transform desired position to body frame
        desired_pos_b, _ = subtract_frame_transforms(
            rob.data.root_pos_w, 
            rob.data.root_quat_w, 
            self._desired_pos_w[:, robot_idx, :]
        )
        
        # Get distance to nearest obstacle
        nearest_obstacle_dist = self._get_nearest_obstacle_distance(rob.data.root_pos_w)  # (num_envs,)
        
        # Get nearest neighbor data (stage-aware: defaults in stages 1-3, actual in stages 4-5)
        neighbor_rel_pos_w, neighbor_rel_vel_w = self._get_nearest_neighbor_data(robot_idx)  # Each: (num_envs, 3)
        
        # Transform neighbor relative data to agent's body frame
        neighbor_rel_pos_b, _ = subtract_frame_transforms(
            rob.data.root_pos_w,
            rob.data.root_quat_w,
            rob.data.root_pos_w + neighbor_rel_pos_w  # Convert relative to absolute, then to body frame
        )
        
        # Relative velocity in body frame
        from isaaclab.utils.math import quat_apply_inverse
        neighbor_rel_vel_b = quat_apply_inverse(rob.data.root_quat_w, neighbor_rel_vel_w)
        
        # Get RM state as one-hot encoding
        # self._rm_states[:, robot_idx] has shape (num_envs,) with values in {0, 1, 2, 3}
        # One-hot encoding: (num_envs, 4)
        rm_state_onehot = torch.nn.functional.one_hot(
            self._rm_states[:, robot_idx], 
            num_classes=self.cfg.reward_cfg.num_rm_states
        ).float()  # (num_envs, 4)
        
        # Concatenate observation components
        obs = torch.cat(
            [
                rob.data.root_lin_vel_b,                # (num_envs, 3) - own velocity
                rob.data.root_ang_vel_b,                # (num_envs, 3) - own angular velocity
                rob.data.projected_gravity_b,           # (num_envs, 3) - gravity direction
                desired_pos_b,                          # (num_envs, 3) - goal position
                nearest_obstacle_dist.unsqueeze(-1),    # (num_envs, 1) - obstacle distance
                neighbor_rel_vel_b,                     # (num_envs, 3) - neighbor relative velocity
                neighbor_rel_pos_b,                     # (num_envs, 3) - neighbor relative position
                rm_state_onehot,                        # (num_envs, 4) - RM state (one-hot)
            ],
            dim=-1,
        )  # -> (num_envs, 23)
        
        return obs

    def _get_observations_slow(self) -> dict:
        """Get observations for each drone including RM state (one-hot encoded).
        
        Neighbor observations behavior:
        - Stages 1-3: Returns neutral defaults (far away, no velocity)
        - Stages 4-5: Returns actual nearest neighbor data
        
        RM state encoding (one-hot):
        - [1, 0, 0, 0]: Hovering (H)
        - [0, 1, 0, 0]: Single-moving (S)
        - [0, 0, 1, 0]: Coop-moving (C)
        - [0, 0, 0, 1]: Obstacle-avoiding (O)
        
        Returns:
            Dictionary mapping agent names directly to observations.
            Format: {"robot_0": obs0, "robot_1": obs1, ...}
        """
        # ✅ UPDATE RM STATES FIRST (before generating observations)
        self._switch_rm_state()
        
        obs_dict = {}
        
        for j in range(self.num_drones):
            agent_name = f"robot_{j}"
            obs_dict[agent_name] = self._build_single_observation(j)
        

        
        if hasattr(self, '_debug_obs_counter'):
            self._debug_obs_counter += 1
        else:
            self._debug_obs_counter = 0
        
        # Print every 100 steps to avoid spam
        # if self._debug_obs_counter % 100 == 0:
        #     agent_0_obs = obs_dict["robot_0"][0]  # First environment, agent 0
        #     agent_0_rm_state = self._rm_states[0, 0].item()  # First environment, agent 0
            
        #     print(f"\n[DEBUG] Agent_0 Observation (env 0, step {self._debug_obs_counter}):")
        #     print(f"  RM State: {agent_0_rm_state} ({self._rm_state_names[agent_0_rm_state]})")
        #     print(f"  Lin Vel (body):       {agent_0_obs[0:3].cpu().numpy()}")
        #     print(f"  Ang Vel (body):       {agent_0_obs[3:6].cpu().numpy()}")
        #     print(f"  Gravity (body):       {agent_0_obs[6:9].cpu().numpy()}")
        #     print(f"  Desired Pos (body):   {agent_0_obs[9:12].cpu().numpy()}")
        #     print(f"  Nearest Obstacle:     {agent_0_obs[12].item():.3f}m")
        #     print(f"  Neighbor Rel Vel:     {agent_0_obs[13:16].cpu().numpy()}")
        #     print(f"  Neighbor Rel Pos:     {agent_0_obs[16:19].cpu().numpy()}")
        #     print(f"  RM State (one-hot):   {agent_0_obs[19:23].cpu().numpy()}")
        #     print(f"  Total obs shape:      {agent_0_obs.shape}")


        return obs_dict
    
    def _get_rewards_slow(self) -> dict[str, torch.Tensor]:
        """Calculate RM state-aware rewards for curriculum learning.
        
        Reward components:
        - Base rewards: velocity penalties, goal progress
        - Formation rewards: cohesion, collision avoidance (stages 4-5)
        - RM state-aware bonuses: altitude control, obstacle avoidance, neighbor coordination
        
        Returns:
            Dictionary mapping agent names to reward tensors of shape (num_envs,)
        """
        # -----------------------------------------------------
        # 1. BASE REWARDS
        # -----------------------------------------------------
        lin_vel_sum = torch.zeros(self.num_envs, device=self.device)
        ang_vel_sum = torch.zeros(self.num_envs, device=self.device)
        dist_goal_sum = torch.zeros(self.num_envs, device=self.device)

        pos_list = []

        # Per-drone base rewards
        for j, rob in enumerate(self._robots):
            lin_vel = torch.sum(torch.square(rob.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(rob.data.root_ang_vel_b), dim=1)
            
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, j, :] - rob.data.root_pos_w, dim=1
            )
            distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

            # ✅ State-dependent scaling for velocity penalties
            # Hovering state: penalize motion more heavily
            # Moving states: penalize less to allow navigation
            is_hovering = (self._rm_states[:, j] == 0).float()  # (num_envs,)
            vel_scale = 1.0 + 0.5 * is_hovering  # 1.5x penalty when hovering, 1.0x otherwise

            lin_vel_sum += lin_vel * vel_scale * self.cfg.reward_cfg.lin_vel_reward_scale * self.step_dt
            ang_vel_sum += ang_vel * vel_scale * self.cfg.reward_cfg.ang_vel_reward_scale * self.step_dt
            dist_goal_sum += distance_to_goal_mapped * self.cfg.reward_cfg.distance_to_goal_reward_scale * self.step_dt

            pos_list.append(rob.data.root_pos_w)

        # -----------------------------------------------------
        # 2. SWARM FORMATION REWARDS (stages 4-5 only)
        # -----------------------------------------------------
        formation_pen = torch.zeros(self.num_envs, device=self.device)
        collision_pen = torch.zeros(self.num_envs, device=self.device)
        
        positions = torch.stack(pos_list, dim=1)  # (num_envs, num_drones, 3)
        
        if self.curriculum_stage in [4, 5]:
            dmat = torch.cdist(positions, positions)  # pairwise distances

            # Formation penalty: too close or too dispersed
            too_close = torch.clamp(self.cfg.swarm_cfg.min_safe_distance - dmat, min=0.0)
            mean_dist = torch.mean(dmat, dim=(1, 2))
            formation_pen = self.cfg.reward_cfg.formation_penalty_scale * (
                torch.mean(too_close, dim=(1, 2)) + 
                torch.clamp(mean_dist - self.cfg.swarm_cfg.max_formation_distance, min=0.0)
            ) * self.step_dt

        # Collision penalty: any drone below minimum height (all stages)
        collided = positions[:, :, 2] < 0.1
        collision_pen = self.cfg.reward_cfg.collision_penalty_scale * collided.any(dim=1).float() * self.step_dt

        # -----------------------------------------------------
        # 3. ✅ NEW: RM STATE-AWARE BONUSES
        # -----------------------------------------------------
        altitude_bonus = torch.zeros(self.num_envs, device=self.device)
        obstacle_bonus = torch.zeros(self.num_envs, device=self.device)
        neighbor_bonus = torch.zeros(self.num_envs, device=self.device)
        state_transition_bonus = torch.zeros(self.num_envs, device=self.device)

        for j, rob in enumerate(self._robots):
            current_state = self._rm_states[:, j]  # (num_envs,)
            agent_z = rob.data.root_pos_w[:, 2]  # (num_envs,)
            
            # -----------------------------------------------
            # 3.1 ALTITUDE CONTROL BONUS (Hovering state)
            # -----------------------------------------------
            # Reward maintaining target altitude when hovering
            is_hovering = (current_state == 0).float()
            target_altitude = self._desired_pos_w[:, j, 2]  # (num_envs,)
            altitude_error = torch.abs(agent_z - target_altitude)
            altitude_bonus_per_agent = torch.exp(-altitude_error) * is_hovering  # Exponential decay with error
            altitude_bonus += altitude_bonus_per_agent * self.cfg.reward_cfg.altitude_bonus_scale * self.step_dt
            
            # -----------------------------------------------
            # 3.2 OBSTACLE AVOIDANCE BONUS (Obstacle-avoiding state)
            # -----------------------------------------------
            # Reward maintaining safe distance from obstacles
            is_avoiding = (current_state == 3).float()
            nearest_obs_dist = self._get_nearest_obstacle_distance(rob.data.root_pos_w)  # (num_envs,)
            
            # Bonus increases with distance (up to safe threshold)
            safe_distance = 2.0  # meters
            obstacle_clearance_ratio = torch.clamp(nearest_obs_dist / safe_distance, 0.0, 1.0)
            obstacle_bonus_per_agent = obstacle_clearance_ratio * is_avoiding
            obstacle_bonus += obstacle_bonus_per_agent * self.cfg.reward_cfg.obstacle_bonus_scale * self.step_dt
            
            # -----------------------------------------------
            # 3.3 NEIGHBOR COORDINATION BONUS (Coop-moving state)
            # -----------------------------------------------
            # Reward maintaining optimal neighbor distance (stages 4-5)
            if self.curriculum_stage in [4, 5]:
                is_cooperating = (current_state == 2).float()
                neighbor_rel_pos, _ = self._get_nearest_neighbor_data(j)
                neighbor_dist = torch.linalg.norm(neighbor_rel_pos, dim=1)  # (num_envs,)
                
                # Optimal neighbor distance: 1.0m (half of max_neighbor_distance)
                optimal_distance = self.cfg.swarm_cfg.max_neighbor_distance / 2.0
                neighbor_error = torch.abs(neighbor_dist - optimal_distance)
                neighbor_bonus_per_agent = torch.exp(-neighbor_error / optimal_distance) * is_cooperating
                neighbor_bonus += neighbor_bonus_per_agent * self.cfg.reward_cfg.neighbor_bonus_scale * self.step_dt
            
            # -----------------------------------------------
            # 3.4 STATE TRANSITION BONUS (Curriculum progression)
            # -----------------------------------------------
            # Small bonus for entering advanced states (encourages progression)
            # Hovering: 0, Single-moving: +0.05, Coop-moving: +0.10, Obstacle-avoiding: +0.15
            state_value = current_state.float() * self.cfg.reward_cfg.state_progress_scale
            state_transition_bonus += state_value * self.step_dt

        # -----------------------------------------------------
        # 4. COMBINE ALL REWARD COMPONENTS
        # -----------------------------------------------------
        # Total reward (shared across all agents in cooperative setting)
        reward = (
            lin_vel_sum +           # Penalty (negative)
            ang_vel_sum +           # Penalty (negative)
            dist_goal_sum +         # Reward (positive)
            formation_pen +         # Penalty (negative)
            collision_pen +         # Penalty (negative)
            altitude_bonus +        # ✅ NEW: Bonus (positive)
            obstacle_bonus +        # ✅ NEW: Bonus (positive)
            neighbor_bonus +        # ✅ NEW: Bonus (positive)
            state_transition_bonus  # ✅ NEW: Bonus (positive)
        )

        # -----------------------------------------------------
        # 5. LOGGING
        # -----------------------------------------------------
        self._episode_sums["lin_vel"] += lin_vel_sum
        self._episode_sums["ang_vel"] += ang_vel_sum
        self._episode_sums["distance_to_goal"] += dist_goal_sum
        self._episode_sums["formation"] += formation_pen
        self._episode_sums["collision"] += collision_pen
        
        # ✅ NEW: Log RM state-aware bonuses
        if "altitude" not in self._episode_sums:
            self._episode_sums["altitude"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums["obstacle_avoid"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums["neighbor_coord"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums["state_progress"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        self._episode_sums["altitude"] += altitude_bonus
        self._episode_sums["obstacle_avoid"] += obstacle_bonus
        self._episode_sums["neighbor_coord"] += neighbor_bonus
        self._episode_sums["state_progress"] += state_transition_bonus

        # Return dictionary with same reward for all agents (cooperative setting)
        return {f"robot_{i}": reward for i in range(self.num_drones)}

#----Termination Conditions with Curriculum Awareness----#  
    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Check termination conditions with curriculum-aware logic.
        
        Termination reasons:
        1. Collision: Any drone too low (< 0.1m) or too high (> max_flight_height)
        2. Out of bounds: Any drone too far from environment origin
        3. Goal reached: All agents reached their goals (stage-dependent)
        4. Timeout: Episode exceeds max_episode_length
        
        Returns:
            Tuple of (terminated_dict, time_out_dict) where each is a dictionary
            mapping agent names to boolean tensors of shape (num_envs,)
        """
        # -----------------------------------------------------
        # 1. TIMEOUT TERMINATION (all stages)
        # -----------------------------------------------------
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # -----------------------------------------------------
        # 2. COLLISION TERMINATION (all stages)
        # -----------------------------------------------------
        # Died if any drone is too low or too high
        died_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for rob in self._robots:
            agent_z = rob.data.root_pos_w[:, 2]  # (num_envs,)
            
            # ✅ IMPROVED: Use configured height bounds
            too_low = agent_z < self.cfg.reward_cfg.min_flight_height  # Below ground/obstacles
            too_high = agent_z > self.cfg.reward_cfg.max_flight_height  # Above safe zone
            
            # Any agent collision causes environment termination
            died_collision = died_collision | too_low | too_high
        
        # -----------------------------------------------------
        # 3. OUT OF BOUNDS TERMINATION (all stages)
        # -----------------------------------------------------
        # Terminate if any drone strays too far from environment origin
        died_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        env_origins = self._terrain.env_origins  # (num_envs, 3)
        max_distance_from_origin = self.cfg.reward_cfg.max_distance_from_origin  # e.g., 50.0m
        
        for rob in self._robots:
            # Calculate XY distance from environment origin
            agent_pos_xy = rob.data.root_pos_w[:, :2]  # (num_envs, 2)
            origin_xy = env_origins[:, :2]  # (num_envs, 2)
            
            distance_from_origin = torch.linalg.norm(agent_pos_xy - origin_xy, dim=1)  # (num_envs,)
            
            # Terminate if any agent too far
            died_out_of_bounds = died_out_of_bounds | (distance_from_origin > max_distance_from_origin)
        
        # -----------------------------------------------------
        # 4. ✅ NEW: GOAL REACHED TERMINATION (curriculum-aware)
        # -----------------------------------------------------
        goal_reached = self._check_goal_reached()  # (num_envs,)
        
        # -----------------------------------------------------
        # 5. COMBINE TERMINATION CONDITIONS
        # -----------------------------------------------------
        # Died = collision OR out of bounds OR goal reached
        died = died_collision | died_out_of_bounds | goal_reached
        
        # -----------------------------------------------------
        # 6. LOGGING (for debugging and analysis)
        # -----------------------------------------------------
        # Store termination reasons for logging in _reset_idx
        self._last_terminated = died
        self._last_timed_out = time_out
        
        # ✅ NEW: Store detailed termination reasons
        if not hasattr(self, '_termination_reasons'):
            self._termination_reasons = {
                'collision': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'out_of_bounds': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'goal_reached': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
        
        self._termination_reasons['collision'] = died_collision
        self._termination_reasons['out_of_bounds'] = died_out_of_bounds
        self._termination_reasons['goal_reached'] = goal_reached
        
        # -----------------------------------------------------
        # 7. RETURN MARL-STYLE DICTIONARIES
        # -----------------------------------------------------
        # All agents share same termination conditions (cooperative setting)
        terminated_dict = {f"robot_{i}": died for i in range(self.num_drones)}
        time_out_dict = {f"robot_{i}": time_out for i in range(self.num_drones)}
        
        return terminated_dict, time_out_dict


    def _check_goal_reached(self) -> torch.Tensor:
        """Check if goals are reached based on current curriculum stage.
        
        Returns:
            Boolean tensor of shape (num_envs,) indicating which environments
            have completed their goals.
        """
        stage = self.curriculum_stage
        
        if stage == 1:
            # Stage 1: Individual hover - all agents within hover threshold
            return self._check_hover_goals_reached()
        
        elif stage == 2:
            # Stage 2: Individual point-to-point - all agents reach their goals
            return self._check_individual_goals_reached()
        
        elif stage == 3:
            # Stage 3: Individual waypoint navigation - all agents complete waypoint paths
            return self._check_waypoint_goals_reached()
        
        elif stage == 4:
            # Stage 4: Swarm navigation - all agents reach swarm goals
            return self._check_swarm_goals_reached()
        
        elif stage == 5:
            # Stage 5: Swarm waypoint navigation - swarm completes waypoint path
            return self._check_swarm_waypoint_goals_reached()
        
        else:
            # Unknown stage - no goal termination
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _check_hover_goals_reached(self) -> torch.Tensor:
        """Check if all agents are hovering at their goal positions.
        
        Returns:
            Boolean tensor (num_envs,) - True if ALL agents within hover threshold
        """
        goal_reached_per_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for j, rob in enumerate(self._robots):
            # Distance to goal
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, j, :] - rob.data.root_pos_w, 
                dim=1
            )  # (num_envs,)
            
            # Velocity magnitude (should be near zero for stable hover)
            velocity_mag = torch.linalg.norm(rob.data.root_lin_vel_w, dim=1)  # (num_envs,)
            
            # Agent reached goal if:
            # 1. Position within threshold (0.15m)
            # 2. Velocity below threshold (0.2 m/s)
            agent_reached = (distance_to_goal < self.cfg.reward_cfg.hover_position_threshold) & \
                        (velocity_mag < self.cfg.reward_cfg.hover_velocity_threshold)
            
            # ALL agents must reach goals
            goal_reached_per_env = goal_reached_per_env & agent_reached
        
        return goal_reached_per_env


    def _check_individual_goals_reached(self) -> torch.Tensor:
        """Check if all agents reached their individual point-to-point goals.
        
        Returns:
            Boolean tensor (num_envs,) - True if ALL agents within goal threshold
        """
        goal_reached_per_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for j, rob in enumerate(self._robots):
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, j, :] - rob.data.root_pos_w, 
                dim=1
            )  # (num_envs,)
            
            # Agent reached goal if within threshold (0.3m for moving goals)
            agent_reached = distance_to_goal < self.cfg.reward_cfg.goal_position_threshold
            
            # ALL agents must reach goals
            goal_reached_per_env = goal_reached_per_env & agent_reached
        
        return goal_reached_per_env


    def _check_waypoint_goals_reached(self) -> torch.Tensor:
        """Check if all agents completed their waypoint paths (Stage 3).
        
        Returns:
            Boolean tensor (num_envs,) - True if ALL agents finished all waypoints
        """
        goal_reached_per_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for j in range(self.num_drones):
            # Check if agent completed all waypoints
            # _current_waypoint_idx >= num_waypoints means completed
            completed = self._current_waypoint_idx[:, j] >= self.num_waypoints_per_agent
            
            # ALL agents must complete waypoints
            goal_reached_per_env = goal_reached_per_env & completed
        
        return goal_reached_per_env


    def _check_swarm_goals_reached(self) -> torch.Tensor:
        """Check if swarm reached goal formation (Stage 4).
        
        Returns:
            Boolean tensor (num_envs,) - True if swarm centroid within threshold
        """
        # Calculate swarm centroid
        swarm_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=1)  # (num_envs, num_drones, 3)
        swarm_centroid = swarm_positions.mean(dim=1)  # (num_envs, 3)
        
        # Calculate goal centroid (average of all agent goals)
        goal_centroid = self._desired_pos_w.mean(dim=1)  # (num_envs, 3)
        
        # Distance from swarm centroid to goal centroid
        centroid_distance = torch.linalg.norm(swarm_centroid - goal_centroid, dim=1)  # (num_envs,)
        
        # Swarm reached goal if centroid within threshold
        goal_reached = centroid_distance < self.cfg.reward_cfg.swarm_goal_threshold  # e.g., 0.5m
        
        return goal_reached


    def _check_swarm_waypoint_goals_reached(self) -> torch.Tensor:
        """Check if swarm completed waypoint path (Stage 5).
        
        Returns:
            Boolean tensor (num_envs,) - True if swarm finished all waypoints
        """
        # Check if swarm completed all waypoints
        # _current_swarm_waypoint_idx >= num_swarm_waypoints means completed
        completed = self._current_swarm_waypoint_idx >= self.num_swarm_waypoints
        
        return completed
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Import at the top of the debug_vis block so it's available everywhere
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG
            
            # ---------------------------------------------------------
            # Goal position markers (yellow spheres)
            # ---------------------------------------------------------
            if not hasattr(self, "goal_pos_visualizers"):
                self.goal_pos_visualizers = []
                for i in range(self.num_drones):
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.05
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 1.0, 0.0)  # yellow
                    marker_cfg.prim_path = f"/Visuals/Command/goal_position_{i}"
                    self.goal_pos_visualizers.append(VisualizationMarkers(marker_cfg))
            
            for viz in self.goal_pos_visualizers:
                viz.set_visibility(True)

            # ---------------------------------------------------------
            # Swarm centroid marker (blue arrow)
            # ---------------------------------------------------------
            if not hasattr(self, "centroid_visualizer"):
                centroid_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                centroid_marker_cfg.prim_path = "/Visuals/Command/swarm_centroid"
                self.centroid_visualizer = VisualizationMarkers(centroid_marker_cfg)

            # Always show centroid (will be hidden in callback if not stages 4/5)
            if hasattr(self, "centroid_visualizer"):
                self.centroid_visualizer.set_visibility(True)

            # ---------------------------------------------------------
            # Stage 3: Individual waypoint markers (green cuboids)
            # ---------------------------------------------------------
            if not hasattr(self, "stage3_waypoint_visualizers"):
                self.stage3_waypoint_visualizers = []
                for i in range(self.num_drones):
                    for wp_idx in range(self.num_waypoints_per_agent):
                        marker_cfg = CUBOID_MARKER_CFG.copy()
                        marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                        marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # green
                        marker_cfg.prim_path = f"/Visuals/Command/stage3_waypoint_agent{i}_wp{wp_idx}"
                        self.stage3_waypoint_visualizers.append(VisualizationMarkers(marker_cfg))
            
            # Don't set visibility here - let _debug_vis_callback handle it based on current stage

            # ---------------------------------------------------------
            # Stage 5: Swarm waypoint markers (green cuboids)
            # ---------------------------------------------------------
            if not hasattr(self, "stage5_waypoint_visualizers"):
                self.stage5_waypoint_visualizers = []
                for wp_idx in range(self.num_swarm_waypoints):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)  # Larger for swarm waypoints
                    marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # green
                    marker_cfg.prim_path = f"/Visuals/Command/stage5_swarm_waypoint_{wp_idx}"
                    self.stage5_waypoint_visualizers.append(VisualizationMarkers(marker_cfg))
            
            # Don't set visibility here - let _debug_vis_callback handle it based on current stage

        # ------------------------------------------------------------------
        # Disable all visualizations
        # ------------------------------------------------------------------
        else:
            if hasattr(self, "goal_pos_visualizers"):
                for viz in self.goal_pos_visualizers:
                    viz.set_visibility(False)

            if hasattr(self, "centroid_visualizer"):
                self.centroid_visualizer.set_visibility(False)

            if hasattr(self, "stage3_waypoint_visualizers"):
                for viz in self.stage3_waypoint_visualizers:
                    viz.set_visibility(False)

            if hasattr(self, "stage5_waypoint_visualizers"):
                for viz in self.stage5_waypoint_visualizers:
                    viz.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        """Update debug visualization markers (matching copy_quadenv.py)."""
        # Update goal position markers (always visible)
        if hasattr(self, "goal_pos_visualizers"):
            for i, viz in enumerate(self.goal_pos_visualizers):
                viz.visualize(self._desired_pos_w[:, i, :])
        
        # Update swarm centroid marker (only for stages 4 and 5)
        if hasattr(self, "centroid_visualizer"):
            if self.curriculum_stage in [4, 5]:
                self.centroid_visualizer.set_visibility(True)
                self.centroid_visualizer.visualize(self._swarm_centroid)
            else:
                self.centroid_visualizer.set_visibility(False)
        
        # Update stage 3 waypoint markers (green cuboids)
        if hasattr(self, "stage3_waypoint_visualizers"):
            if self.curriculum_stage == 3:
                # Show and update waypoints
                viz_idx = 0
                for agent_idx in range(self.num_drones):
                    for wp_idx in range(self.num_waypoints_per_agent):
                        # Get waypoint positions for all environments: (num_envs, 3)
                        waypoint_positions = self._waypoint_paths[:, agent_idx, wp_idx, :]
                        self.stage3_waypoint_visualizers[viz_idx].set_visibility(True)
                        self.stage3_waypoint_visualizers[viz_idx].visualize(waypoint_positions)
                        viz_idx += 1
            else:
                # Hide all stage 3 waypoints
                for viz in self.stage3_waypoint_visualizers:
                    viz.set_visibility(False)
        
        # Update stage 5 swarm waypoint markers (green cuboids)
        if hasattr(self, "stage5_waypoint_visualizers"):
            if self.curriculum_stage == 5:
                # Show and update waypoints
                for wp_idx in range(self.num_swarm_waypoints):
                    # Get swarm waypoint positions for all environments: (num_envs, 3)
                    waypoint_positions = self._swarm_waypoint_paths[:, wp_idx, :]
                    self.stage5_waypoint_visualizers[wp_idx].set_visibility(True)
                    self.stage5_waypoint_visualizers[wp_idx].visualize(waypoint_positions)
            else:
                # Hide all stage 5 waypoints
                for viz in self.stage5_waypoint_visualizers:
                    viz.set_visibility(False)
    
    
            
###--- Reward Machine helper methods ---###
    def _switch_rm_state(self):
        """Update Reward Machine states for all agents based on current conditions (VECTORIZED).
        
        State transitions:
        - Hovering (0): agent_z <= hover_min_altitude
        - Single-moving (1): agent_z > hover_min AND obstacle_dist > threshold AND neighbor_dist >= max_neighbor_distance
        - Coop-moving (2): agent_z > hover_min AND obstacle_dist > threshold AND neighbor_dist < max_neighbor_distance
        - Obstacle-avoiding (3): agent_z > hover_min AND obstacle_dist <= threshold
        
        Updates self._rm_states: (num_envs, num_drones) tensor with state indices
        """
        # ✅ VECTORIZED: Stack all robot positions at once
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)  # (num_drones, num_envs, 3)
        
        # Extract altitudes: (num_drones, num_envs)
        all_z = all_positions[:, :, 2]
        
        # Get obstacle distances (vectorized for all agents): (num_drones, num_envs)
        nearest_obstacle_dists = self._get_nearest_obstacle_distance_vectorized(all_positions)
        
        # Get neighbor distances (vectorized for all agents)
        # We need only distances, not full relative positions
        if self.curriculum_stage in [1, 2, 3]:
            # Default: all agents far from neighbors
            neighbor_dists = torch.full(
                (self.num_drones, self.num_envs), 
                self.cfg.swarm_cfg.max_neighbor_distance, 
                device=self.device
            )
        else:
            # Stages 4-5: Compute actual pairwise distances
            # all_positions: (num_drones, num_envs, 3)
            diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)  # (num_drones, num_drones, num_envs, 3)
            distances = torch.linalg.norm(diff, dim=3)  # (num_drones, num_drones, num_envs)
            
            # Set self-distances to large value
            eye_mask = torch.eye(self.num_drones, device=self.device).unsqueeze(2)  # (num_drones, num_drones, 1)
            distances = distances + eye_mask * 1e6
            
            # Find nearest neighbor distance for each agent: (num_drones, num_envs)
            neighbor_dists = distances.min(dim=1)[0]
        
        # ✅ VECTORIZED STATE TRANSITION LOGIC
        # All operations on (num_drones, num_envs) tensors
        
        # Initialize all states as Hovering (0)
        new_states = torch.zeros_like(all_z, dtype=torch.long)  # (num_drones, num_envs)
        
        # Check conditions
        above_hover = all_z > self.cfg.reward_cfg.hover_min_altitude
        far_from_obstacle = nearest_obstacle_dists > self.cfg.reward_cfg.close_obs_dist_thresh
        near_neighbor = neighbor_dists < self.cfg.swarm_cfg.max_neighbor_distance
        
        # Apply state transitions (order matters - later assignments override earlier ones)
        # State 1 (S): Above hover AND far from obstacle AND far from neighbor
        single_moving_mask = above_hover & far_from_obstacle & (~near_neighbor)
        new_states[single_moving_mask] = 1
        
        # State 2 (C): Above hover AND far from obstacle AND near neighbor
        coop_moving_mask = above_hover & far_from_obstacle & near_neighbor
        new_states[coop_moving_mask] = 2
        
        # State 3 (O): Above hover AND close to obstacle (overrides states 1 & 2)
        obstacle_avoiding_mask = above_hover & (~far_from_obstacle)
        new_states[obstacle_avoiding_mask] = 3
        
        # State 0 (H): Below hover threshold (overrides all - highest priority)
        hovering_mask = ~above_hover
        new_states[hovering_mask] = 0
        
        # ✅ Update state buffer: (num_drones, num_envs) -> transpose -> (num_envs, num_drones)
        self._rm_states = new_states.transpose(0, 1)
            
    def _compute_altitude_bonus_vectorized(self, all_positions, desired_transposed):
        """Vectorized altitude bonus computation."""
        is_hovering = (self._rm_states.transpose(0, 1) == 0).float()  # (num_drones, num_envs)
        target_altitude = desired_transposed[:, :, 2]  # (num_drones, num_envs)
        altitude_error = torch.abs(all_positions[:, :, 2] - target_altitude)
        bonus = (torch.exp(-altitude_error) * is_hovering).sum(dim=0)
        return bonus * self.cfg.reward_cfg.altitude_bonus_scale * self.step_dt

    def _get_nearest_obstacle_distance_vectorized(self, all_positions: torch.Tensor) -> torch.Tensor:
        """Vectorized obstacle distance calculation for all agents.
        
        Args:
            all_positions: Agent positions, shape (num_drones, num_envs, 3)
        
        Returns:
            Nearest obstacle distances, shape (num_drones, num_envs)
            Clamped to [0, max_obstacle_distance]
        """
        if self._obstacle_positions is None or self.curriculum_stage not in [3, 5]:
            # No obstacles active in current stage
            return torch.full(
                (self.num_drones, self.num_envs), 
                self.cfg.curriculum.max_obstacle_distance, 
                device=self.device
            )
        
        # Reshape for broadcasting
        # all_positions: (num_drones, num_envs, 3)
        # obstacle_positions: (num_obstacles, 3)
        # Result after broadcasting: (num_drones, num_envs, num_obstacles, 3)
        
        # Expand dimensions for broadcasting
        agent_pos_expanded = all_positions.unsqueeze(2)  # (num_drones, num_envs, 1, 3)
        obs_pos_expanded = self._obstacle_positions.unsqueeze(0).unsqueeze(0)  # (1, 1, num_obstacles, 3)
        
        # Calculate distances: (num_drones, num_envs, num_obstacles)
        diff = agent_pos_expanded - obs_pos_expanded
        distances = torch.linalg.norm(diff, dim=3)  # (num_drones, num_envs, num_obstacles)
        
        # Find minimum distance to any obstacle: (num_drones, num_envs)
        min_distances = distances.min(dim=2)[0]
        
        # Clamp to maximum range
        min_distances = torch.clamp(min_distances, 0.0, self.cfg.curriculum.max_obstacle_distance)
        
        return min_distances

    def _compute_obstacle_bonus_vectorized(self, all_positions: torch.Tensor) -> torch.Tensor:
        """Vectorized obstacle avoidance bonus computation.
        
        Args:
            all_positions: Agent positions, shape (num_drones, num_envs, 3)
        
        Returns:
            Obstacle bonus summed across all agents, shape (num_envs,)
        """
        # Get obstacle-avoiding mask: (num_drones, num_envs)
        is_avoiding = (self._rm_states.transpose(0, 1) == 3).float()
        
        # Get nearest obstacle distances: (num_drones, num_envs)
        nearest_obs_dist = self._get_nearest_obstacle_distance_vectorized(all_positions)
        
        # Bonus increases with distance (up to safe threshold)
        safe_distance = 2.0  # meters
        obstacle_clearance_ratio = torch.clamp(nearest_obs_dist / safe_distance, 0.0, 1.0)
        
        # Apply bonus only when in obstacle-avoiding state: (num_drones, num_envs)
        bonus_per_agent = obstacle_clearance_ratio * is_avoiding
        
        # Sum across all agents: (num_envs,)
        bonus = bonus_per_agent.sum(dim=0)
        
        return bonus * self.cfg.reward_cfg.obstacle_bonus_scale * self.step_dt

    def _compute_neighbor_bonus_vectorized(self, all_positions: torch.Tensor) -> torch.Tensor:
        """Vectorized neighbor coordination bonus computation.
        
        Args:
            all_positions: Agent positions, shape (num_drones, num_envs, 3)
        
        Returns:
            Neighbor bonus summed across all agents, shape (num_envs,)
        """
        # Only compute for swarm stages
        if self.curriculum_stage not in [4, 5]:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Get cooperating mask: (num_drones, num_envs)
        is_cooperating = (self._rm_states.transpose(0, 1) == 2).float()
        
        # Compute pairwise distances: (num_drones, num_drones, num_envs)
        diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)  # (num_drones, num_drones, num_envs, 3)
        distances = torch.linalg.norm(diff, dim=3)  # (num_drones, num_drones, num_envs)
        
        # Set self-distances to large value to ignore
        eye_mask = torch.eye(self.num_drones, device=self.device).unsqueeze(2)  # (num_drones, num_drones, 1)
        distances = distances + eye_mask * 1e6
        
        # Find nearest neighbor distance for each agent: (num_drones, num_envs)
        neighbor_dist = distances.min(dim=1)[0]  # (num_drones, num_envs)
        
        # Optimal neighbor distance
        optimal_distance = self.cfg.reward_cfg.optimal_neighbor_distance
        
        # Calculate bonus based on distance error
        neighbor_error = torch.abs(neighbor_dist - optimal_distance)
        bonus_per_agent = torch.exp(-neighbor_error / optimal_distance) * is_cooperating
        
        # Sum across all agents: (num_envs,)
        bonus = bonus_per_agent.sum(dim=0)
        
        return bonus * self.cfg.reward_cfg.neighbor_bonus_scale * self.step_dt

    def _compute_state_bonus_vectorized(self) -> torch.Tensor:
        """Vectorized state progression bonus computation.
        
        Returns:
            State bonus summed across all agents, shape (num_envs,)
        """
        # _rm_states: (num_envs, num_drones) -> transpose -> (num_drones, num_envs)
        rm_states_transposed = self._rm_states.transpose(0, 1).float()
        
        # State value: Hovering=0, Single-moving=1, Coop-moving=2, Obstacle-avoiding=3
        # Multiply by scale to get bonus
        state_value_per_agent = rm_states_transposed * self.cfg.reward_cfg.state_progress_scale
        
        # Sum across all agents: (num_envs,)
        bonus = state_value_per_agent.sum(dim=0)
        
        return bonus * self.step_dt


###------ Distance based helpers ------###
    def _get_nearest_obstacle_distance(self, agent_position: torch.Tensor) -> torch.Tensor:
        """Calculate distance from agent to nearest obstacle.
        
        Args:
            agent_position: Position of the agent, shape (num_envs, 3)
        
        Returns:
            Distance to nearest obstacle, shape (num_envs,)
            Clamped to [0, max_obstacle_distance]
        """
        if self._obstacle_positions is None or self.curriculum_stage not in [3, 5]:
            # No obstacles active in current stage
            return torch.full((self.num_envs,), self.cfg.curriculum.max_obstacle_distance, device=self.device)
        
        # Calculate distances to all obstacles
        # agent_position: (num_envs, 3)
        # obstacle_positions: (num_obstacles, 3)
        # Result: (num_envs, num_obstacles)
        distances = torch.cdist(agent_position.unsqueeze(1), self._obstacle_positions.unsqueeze(0)).squeeze(1)
        
        # Find minimum distance to any obstacle
        min_distances, _ = distances.min(dim=-1)  # (num_envs,)
        
        # Clamp to maximum range
        min_distances = torch.clamp(min_distances, 0.0, self.cfg.curriculum.max_obstacle_distance)
        
        return min_distances

    def _get_nearest_neighbor_data(self, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate relative position and velocity to nearest neighbor for each agent.
        
        For stages 1-3 (individual training), returns neutral default values:
        - Relative position: (max_neighbor_distance, 0, 0) - indicates "far away, no interaction needed"
        - Relative velocity: (0, 0, 0) - indicates "no relative motion"
        
        For stages 4-5 (swarm training), returns actual nearest neighbor data.
        
        Args:
            agent_idx: Index of the agent (0 to num_drones-1)
        
        Returns:
            Tuple of (relative_position, relative_velocity) where:
            - relative_position: (num_envs, 3) - position of nearest neighbor relative to agent in world frame
            - relative_velocity: (num_envs, 3) - velocity of nearest neighbor relative to agent in world frame
        """
        # ✅ Check if we're in individual training stages (1, 2, 3)
        if self.curriculum_stage in [1, 2, 3]:
            # Return neutral defaults indicating "no neighbor interaction needed"
            # Position: far away in X direction (max_neighbor_distance, 0, 0)
            # Velocity: zero relative motion
            default_rel_pos = torch.zeros(self.num_envs, 3, device=self.device)
            default_rel_pos[:, 0] = self.cfg.swarm_cfg.max_neighbor_distance  # Far away in X
            
            default_rel_vel = torch.zeros(self.num_envs, 3, device=self.device)
            
            return default_rel_pos, default_rel_vel
        
        # ✅ Stages 4-5: Compute actual nearest neighbor data for swarm coordination
        
        # Get current agent's data
        current_agent = self._robots[agent_idx]
        current_pos = current_agent.data.root_pos_w  # (num_envs, 3)
        current_vel = current_agent.data.root_lin_vel_w  # (num_envs, 3)
        
        # Collect all other agents' positions and velocities
        other_positions = []  # Will be (num_envs, num_other_agents, 3)
        other_velocities = []  # Will be (num_envs, num_other_agents, 3)
        
        for j, rob in enumerate(self._robots):
            if j != agent_idx:  # Exclude self
                other_positions.append(rob.data.root_pos_w)
                other_velocities.append(rob.data.root_lin_vel_w)
        
        if len(other_positions) == 0:
            # Only one agent in swarm - return neutral defaults
            default_rel_pos = torch.zeros(self.num_envs, 3, device=self.device)
            default_rel_pos[:, 0] = self.cfg.max_neighbor_distance
            
            default_rel_vel = torch.zeros(self.num_envs, 3, device=self.device)
            
            return default_rel_pos, default_rel_vel
        
        # Stack other agents' data: (num_envs, num_other_agents, 3)
        other_positions = torch.stack(other_positions, dim=1)
        other_velocities = torch.stack(other_velocities, dim=1)
        
        # Calculate distances to all other agents
        # current_pos: (num_envs, 3) -> (num_envs, 1, 3)
        # other_positions: (num_envs, num_other_agents, 3)
        # distances: (num_envs, num_other_agents)
        distances = torch.linalg.norm(
            other_positions - current_pos.unsqueeze(1), 
            dim=2
        )
        
        # Find nearest neighbor for each environment
        # nearest_idx: (num_envs,) - indices of nearest neighbors
        nearest_idx = torch.argmin(distances, dim=1)
        
        # Gather nearest neighbor positions and velocities
        # Create index tensor for gathering: (num_envs, 1, 3)
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).unsqueeze(2)
        neighbor_indices = nearest_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)
        
        # Gather: (num_envs, 1, 3) -> squeeze -> (num_envs, 3)
        nearest_pos = torch.gather(other_positions, 1, neighbor_indices).squeeze(1)
        nearest_vel = torch.gather(other_velocities, 1, neighbor_indices).squeeze(1)
        
        # Calculate relative quantities (in world frame)
        relative_position = nearest_pos - current_pos  # (num_envs, 3)
        relative_velocity = nearest_vel - current_vel  # (num_envs, 3)
        
        # ✅ OPTIONAL: Clamp relative position magnitude to max_neighbor_distance
        # This prevents extremely large values if agents are far apart
        rel_pos_norm = torch.linalg.norm(relative_position, dim=1, keepdim=True)
        relative_position = torch.where(
            rel_pos_norm > self.cfg.swarm_cfg.max_neighbor_distance,
            relative_position * (self.cfg.swarm_cfg.max_neighbor_distance / (rel_pos_norm + 1e-8)),
            relative_position
        )
        
        return relative_position, relative_velocity

    def _get_nearest_neighbor_data_vectorized(
    self, 
    all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
    all_quats: torch.Tensor       # (num_drones, num_envs, 4)
) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized nearest neighbor calculation for all agents simultaneously.
        
        Returns:
            neighbor_rel_pos_b: (num_drones, num_envs, 3) - relative positions in body frame
            neighbor_rel_vel_b: (num_drones, num_envs, 3) - relative velocities in body frame
        """
        if self.curriculum_stage in [1, 2, 3]:
            # Return defaults
            default_rel_pos = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
            default_rel_pos[:, :, 0] = self.cfg.swarm_cfg.max_neighbor_distance
            default_rel_vel = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
            return default_rel_pos, default_rel_vel
        
        # Compute pairwise distances: (num_drones, num_drones, num_envs)
        diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)  # (num_drones, num_drones, num_envs, 3)
        distances = torch.linalg.norm(diff, dim=3)  # (num_drones, num_drones, num_envs)
        
        # Set self-distances to large value to ignore
        eye_mask = torch.eye(self.num_drones, device=self.device).unsqueeze(2)  # (num_drones, num_drones, 1)
        distances = distances + eye_mask * 1e6
        
        # Find nearest neighbor for each agent: (num_drones, num_envs)
        nearest_idx = torch.argmin(distances, dim=1)  # (num_drones, num_envs)
        
        # ✅ CORRECT GATHERING using batch indexing
        # Create indices for gathering
        batch_idx = torch.arange(self.num_envs, device=self.device).view(1, -1)  # (1, num_envs)
        drone_idx = torch.arange(self.num_drones, device=self.device).view(-1, 1)  # (num_drones, 1)
        
        # Gather nearest neighbor positions
        # all_positions: (num_drones, num_envs, 3)
        # nearest_idx: (num_drones, num_envs) - indices in range [0, num_drones-1]
        # We want: all_positions[nearest_idx[i, j], j, :] for each (i, j)
        
        nearest_pos = all_positions[
            nearest_idx,  # (num_drones, num_envs) - selects which drone
            batch_idx.expand(self.num_drones, -1)  # (num_drones, num_envs) - selects which env
        ]  # Result: (num_drones, num_envs, 3)
        
        # Relative position (world frame)
        relative_pos_w = nearest_pos - all_positions  # (num_drones, num_envs, 3)
        
        # Clamp magnitude
        rel_pos_norm = torch.linalg.norm(relative_pos_w, dim=2, keepdim=True)  # (num_drones, num_envs, 1)
        relative_pos_w = torch.where(
            rel_pos_norm > self.cfg.swarm_cfg.max_neighbor_distance,
            relative_pos_w * (self.cfg.swarm_cfg.max_neighbor_distance / (rel_pos_norm + 1e-8)),
            relative_pos_w
        )
        
        # Transform to body frame (vectorized)
        from isaaclab.utils.math import quat_apply_inverse
        relative_pos_b = quat_apply_inverse(
            all_quats.reshape(-1, 4),
            relative_pos_w.reshape(-1, 3)
        ).reshape(self.num_drones, self.num_envs, 3)
        
        # ✅ SAME PROCESS FOR VELOCITY
        all_lin_vels = torch.stack([rob.data.root_lin_vel_w for rob in self._robots], dim=0)  # (num_drones, num_envs, 3)
        
        nearest_vel = all_lin_vels[
            nearest_idx,  # (num_drones, num_envs)
            batch_idx.expand(self.num_drones, -1)  # (num_drones, num_envs)
        ]  # Result: (num_drones, num_envs, 3)
        
        relative_vel_w = nearest_vel - all_lin_vels
        relative_vel_b = quat_apply_inverse(
            all_quats.reshape(-1, 4),
            relative_vel_w.reshape(-1, 3)
        ).reshape(self.num_drones, self.num_envs, 3)
        
        return relative_pos_b, relative_vel_b



###---------- Curriculum Methods ----------###
    
    def _update_waypoint_goals_slow(self):
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
                        
                        #print(f"[Stage 3] Env {env_idx}, Agent {agent_idx}: Reached waypoint {current_wp_idx}, advancing to {next_wp_idx}")
                    else:
                        # Mark as completed (stay at last waypoint)
                        self._current_waypoint_idx[env_idx, agent_idx] = self.num_waypoints_per_agent
                        #print(f"[Stage 3] Env {env_idx}, Agent {agent_idx}: Completed all waypoints!")
    
    def _update_waypoint_goals(self):
        """Vectorized waypoint update for all environments and agents."""
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_positions = all_positions.transpose(0, 1)  # (num_envs, num_drones, 3)
        
        current_wp_idx = self._current_waypoint_idx  # (num_envs, num_drones)
        not_finished = current_wp_idx < self.num_waypoints_per_agent
        
        # Gather current waypoints
        env_idx = torch.arange(self.num_envs, device=self.device).view(-1, 1, 1)
        drone_idx = torch.arange(self.num_drones, device=self.device).view(1, -1, 1)
        wp_idx = current_wp_idx.unsqueeze(2).clamp(max=self.num_waypoints_per_agent - 1)
        coord_idx = torch.arange(3, device=self.device).view(1, 1, -1)
        
        current_waypoints = self._waypoint_paths[env_idx, drone_idx, wp_idx, coord_idx]
        distances = torch.linalg.norm(all_positions - current_waypoints, dim=2)
        
        # Check which agents reached waypoints
        reached = (distances < self.waypoint_reach_threshold) & not_finished
        
        # ✅ INCREMENT with clamping to max
        self._current_waypoint_idx = torch.where(
            reached,
            torch.clamp(current_wp_idx + 1, max=self.num_waypoints_per_agent),  # ✅ Explicit cap
            current_wp_idx
        )
        
        # Update goals
        next_wp_idx = self._current_waypoint_idx.unsqueeze(2).clamp(max=self.num_waypoints_per_agent - 1)
        next_waypoints = self._waypoint_paths[env_idx, drone_idx, next_wp_idx, coord_idx]
        
        self._desired_pos_w = torch.where(
            reached.unsqueeze(2).expand(-1, -1, 3),
            next_waypoints,
            self._desired_pos_w
        )

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
                    
                    #print(f"[Stage 5] Env {env_idx}: Swarm reached waypoint {current_wp_idx}, advancing to {next_wp_idx}")
                else:
                    # Mark as completed
                    self._current_swarm_waypoint_idx[env_idx] = self.num_swarm_waypoints
                    #print(f"[Stage 5] Env {env_idx}: Swarm completed all waypoints!")

    def _build_all_obstacles(self):
        """Build all obstacle prims at fixed locations for stages 3 and 5."""
        self._build_stage3_obstacles()
        self._build_stage5_obstacles()
        self._obstacles_built = True
        
        # ✅ ADDED: Store obstacle positions for distance calculations
        self._collect_obstacle_positions()

    def _collect_obstacle_positions(self):
        """Collect all obstacle center positions into a tensor for distance calculations."""
        obstacle_positions = []
        
        # Stage 3 obstacles: num_drones * 3
        stage3_offset = (0,0, 0.0)
        obstacle_size = (0.2, 0.8, 5.0)
        base_height = 2.5
        lane_width = 1.5
        obstacle_spacing_x = 3.0
        lateral_offset = 0.4
        course_start_x = stage3_offset[0] + 3.0
        
        for agent_idx in range(self.num_drones):
            lane_center_y = stage3_offset[1] + (agent_idx - (self.num_drones - 1) / 2.0) * lane_width
            
            for obs_idx in range(3):
                obs_x = course_start_x + obs_idx * obstacle_spacing_x
                
                if obs_idx == 0:
                    obs_y = lane_center_y
                elif obs_idx == 1:
                    obs_y = lane_center_y - lateral_offset
                else:
                    obs_y = lane_center_y + lateral_offset
                
                obs_z = base_height
                obstacle_positions.append([obs_x, obs_y, obs_z])
        
        # Stage 5 obstacles: 8 walls
        stage5_offset = (0, 0, 0.0)
        x_offset = self.cfg.curriculum.stage5_obsx_offset
        y_offset = self.cfg.curriculum.stage5_obsy_offset
        base_height = 2.5
        dist_from_spawn_swarm = self.cfg.curriculum.dist_from_spawn_swarm
        base_y = stage5_offset[1] + dist_from_spawn_swarm
        base_x = stage5_offset[0]
        
        wall_positions = [
            (base_x - x_offset, base_y, base_height),
            (base_x + x_offset, base_y, base_height),
            (base_x, base_y + 0.5 * y_offset, base_height),
            (base_x - x_offset, base_y + y_offset, base_height),
            (base_x + x_offset, base_y + y_offset, base_height),
            (base_x, base_y + 1.5 * y_offset, base_height),
            (base_x - x_offset, base_y + 2 * y_offset, base_height),
            (base_x + x_offset, base_y + 2 * y_offset, base_height),
        ]
        
        obstacle_positions.extend(wall_positions)
        
        # Convert to tensor: (num_obstacles, 3)
        self._obstacle_positions = torch.tensor(obstacle_positions, dtype=torch.float32, device=self.device)
        
        #print(f"[INFO] Collected {len(obstacle_positions)} obstacle positions for distance calculations")
        
    def _build_stage3_obstacles(self):
        """Build stage 3 obstacle course: 3 obstacles per agent in zig-zag pattern.
        
        Each agent gets 3 obstacles arranged in a zig-zag pattern:
        - Obstacle 1: Centered in agent's lane
        - Obstacle 2: Offset to left
        - Obstacle 3: Offset to right
        
        Obstacles are indexed from 0 to (num_agents * 3 - 1) for unified RayCaster detection.
        """
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import (
            RigidBodyPropertiesCfg,
            CollisionPropertiesCfg,
        )
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        stage3_offset = (0, 0, 0.0)
        source_env_idx = 0
        
        # Obstacle course parameters
        obstacle_size = (0.2, 0.8, 15)  # (thickness, width, height)
        base_height = 7.5  # Half of obstacle height
        
        # Lane configuration
        lane_width = 1.5  # Width of each agent's lane
        obstacle_spacing_x = 3.0  # Distance between obstacles along X-axis
        lateral_offset = 0.4  # How far obstacles shift left/right
        
        # Starting X position for obstacle course
        course_start_x = stage3_offset[0] + 3.0
        
        # Global obstacle index counter (starts at 0 for stage 3)
        global_wall_idx = 0
        
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
                
                # Create obstacle prim with unified naming scheme
                wall_path = f"/World/envs/env_{source_env_idx}/obstacles/wall_{global_wall_idx}"
                wall_cfg = CuboidCfg(
                    size=obstacle_size,
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.1, 0.1),  # Red for stage 3
                        roughness=0.4,
                        metallic=0.0,
                    ),
                )
                wall_cfg.func(wall_path, wall_cfg, translation=(obs_x, obs_y, obs_z))
                
                # Increment global index
                global_wall_idx += 1
        
        #print(f"[INFO] Built Stage 3 obstacle course:")
        #print(f"  - {self.num_drones} agents × 3 obstacles = {self.num_drones * 3} total obstacles")
        #print(f"  - Obstacle indices: 0 to {self.num_drones * 3 - 1}")
        #print(f"  - Zig-zag pattern with {obstacle_spacing_x}m spacing")

    def _build_stage5_obstacles(self):
        """Build stage 5 obstacles in stacked X pattern.
        
        Creates 8 wall segments arranged in two X shapes stacked vertically:
        - Bottom X: 3 walls (left, center, right)
        - Top X: 5 walls (full X pattern)
        
        Obstacles are indexed starting from (num_agents * 3) to allow unified RayCaster detection.
        """
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import (
            RigidBodyPropertiesCfg,
            CollisionPropertiesCfg,
        )
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        stage5_offset = (0, 0, 0.0)
        source_env_idx = 0
        
        # Get offset configuration
        x_offset = self.cfg.curriculum.stage5_obsx_offset  # 2.0m
        y_offset = self.cfg.curriculum.stage5_obsy_offset  # 3.0m
        
        # Obstacle size: vertical walls blocking Y-axis travel
        obstacle_size = (1.2, 0.2, 15)  # (length, thickness, height)
        base_height = 7.5
        
        dist_from_spawn_swarm = self.cfg.curriculum.dist_from_spawn_swarm
        
        # Base Y position for the pattern
        base_y = stage5_offset[1] + dist_from_spawn_swarm
        base_x = stage5_offset[0]
        
        # Calculate wall positions based on stacked X pattern
        wall_positions = [
            # Bottom X base (3 walls)
            (base_x - x_offset, base_y, base_height),              # wall0 - left bottom
            (base_x + x_offset, base_y, base_height),              # wall1 - right bottom
            (base_x, base_y + 0.5 * y_offset, base_height),        # wall2 - center bottom
            
            # Middle layer (2 walls)
            (base_x - x_offset, base_y + y_offset, base_height),   # wall3 - left middle
            (base_x + x_offset, base_y + y_offset, base_height),   # wall4 - right middle
            
            # Top X (3 walls)
            (base_x, base_y + 1.5 * y_offset, base_height),        # wall5 - center top
            (base_x - x_offset, base_y + 2 * y_offset, base_height), # wall6 - left top
            (base_x + x_offset, base_y + 2 * y_offset, base_height), # wall7 - right top
        ]
        
        # Global obstacle index starts after stage 3 obstacles
        stage5_start_idx = self.num_drones * 3
        
        # Build all 8 wall segments
        for local_idx, (wall_x, wall_y, wall_z) in enumerate(wall_positions):
            global_wall_idx = stage5_start_idx + local_idx
            
            wall_path = f"/World/envs/env_{source_env_idx}/obstacles/wall_{global_wall_idx}"
            wall_cfg = CuboidCfg(
                size=obstacle_size,
                rigid_props=RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.9),  # Blue for stage 5
                    roughness=0.4,
                    metallic=0.0,
                ),
            )
            wall_cfg.func(wall_path, wall_cfg, translation=(wall_x, wall_y, wall_z))
        
        #print(f"[INFO] Built Stage 5 stacked X obstacle pattern:")
        #print(f"  - 8 wall segments in X formation")
        #print(f"  - Obstacle indices: {stage5_start_idx} to {stage5_start_idx + 7}")
        #print(f"  - X offset: {x_offset}m, Y offset: {y_offset}m")
        #print(f"  - Base position: ({base_x}, {base_y})")
    

    def _collect_obstacle_positions_stage3(self):
        """Collect Stage 3 obstacle positions using shared parameters."""
        obstacle_positions = []
        
        # ✅ GET SHARED PARAMETERS from config
        params = self.cfg.curriculum.get_stage3_params()
        obstacle_size = self.cfg.curriculum.obstacles_size
        base_height = obstacle_size[2] / 2.0
        
        for agent_idx in range(self.num_drones):
            lane_center_y = (agent_idx - (self.num_drones - 1) / 2.0) * params["lane_width"]
            
            for obs_idx in range(3):
                obs_x = params["course_start_x"] + obs_idx * params["obstacle_spacing_x"]
                
                if obs_idx == 0:
                    obs_y = lane_center_y
                elif obs_idx == 1:
                    obs_y = lane_center_y - params["lateral_offset"]
                else:
                    obs_y = lane_center_y + params["lateral_offset"]
                
                obs_z = base_height
                obstacle_positions.append([obs_x, obs_y, obs_z])
        
        self._obstacle_positions = torch.tensor(
            obstacle_positions, 
            dtype=torch.float32, 
            device=self.device
        )
        
        print(f"[INFO] Collected {len(obstacle_positions)} Stage 3 obstacle positions")

    def _collect_obstacle_positions_stage5(self):
        """Collect Stage 5 obstacle positions matching REDUCED parameters."""
        obstacle_positions = []
        
        # ✅ MATCH _build_stage5_obstacles_at_origin() parameters
        x_offset = self.cfg.curriculum.stage5_obsx_offset  # 1.5m
        y_offset = self.cfg.curriculum.stage5_obsy_offset  # 2.0m
        base_height = 4.0  # ✅ REDUCED from 2.5
        dist_from_spawn_swarm = self.cfg.curriculum.dist_from_spawn_swarm
        base_y = dist_from_spawn_swarm  # ✅ No Y offset
        base_x = 0.0  # ✅ At origin
        
        wall_positions = [
            (base_x - x_offset, base_y, base_height),
            (base_x + x_offset, base_y, base_height),
            (base_x, base_y + 0.5 * y_offset, base_height),
            (base_x - x_offset, base_y + y_offset, base_height),
            (base_x + x_offset, base_y + y_offset, base_height),
            (base_x, base_y + 1.5 * y_offset, base_height),
            (base_x - x_offset, base_y + 2 * y_offset, base_height),
            (base_x + x_offset, base_y + 2 * y_offset, base_height),
        ]
        
        obstacle_positions.extend(wall_positions)
        
        # Convert to tensor and store
        self._obstacle_positions = torch.tensor(
            obstacle_positions, 
            dtype=torch.float32, 
            device=self.device
        )
        
        #print(f"[INFO] Collected {len(obstacle_positions)} Stage 5 obstacle positions")

    # ✅ NEW: Obstacle builders at ORIGIN (no offsets)
    def _build_stage3_obstacles_at_origin(self):
        """Build stage 3 obstacles at environment origin (0, 0)."""
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        source_env_idx = 0
        
        # ✅ GET SHARED PARAMETERS from config
        params = self.cfg.curriculum.get_stage3_params()
        obstacle_size = self.cfg.curriculum.obstacles_size
        base_height = obstacle_size[2] / 2.0  # Half of obstacle height
        
        global_wall_idx = 0
        
        for agent_idx in range(self.num_drones):
            lane_center_y = (agent_idx - (self.num_drones - 1) / 2.0) * params["lane_width"]
            
            for obs_idx in range(3):
                obs_x = params["course_start_x"] + obs_idx * params["obstacle_spacing_x"]
                
                # Zig-zag pattern
                if obs_idx == 0:
                    obs_y = lane_center_y
                elif obs_idx == 1:
                    obs_y = lane_center_y - params["lateral_offset"]
                else:
                    obs_y = lane_center_y + params["lateral_offset"]
                
                obs_z = base_height
                
                wall_path = f"/World/envs/env_{source_env_idx}/obstacles/wall_{global_wall_idx}"
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
                global_wall_idx += 1
        
        # Collect obstacle positions
        self._collect_obstacle_positions_stage3()
        
        #print(f"[INFO] Built {global_wall_idx} Stage 3 obstacles using shared parameters")
    
    
    def _build_stage5_obstacles_at_origin(self):
        """Build stage 5 obstacles at environment origin (0, 0)."""
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        source_env_idx = 0
        
        # ✅ REDUCED: Obstacle parameters
        x_offset = self.cfg.curriculum.stage5_obsx_offset  # 1.5m
        y_offset = self.cfg.curriculum.stage5_obsy_offset  # 2.0m
        obstacle_size = self.cfg.curriculum.obstacles_size  # ✅ Reduced from (1.2, 0.2, 15)
        #Inverted dimensions for vertical walls related to Y-axis travel
        obstacle_size = (obstacle_size[1], obstacle_size[0], obstacle_size[2])  # Reduced height
        base_height = obstacle_size[2] / 2.0  # Half of obstacle height
        
        dist_from_spawn_swarm = self.cfg.curriculum.dist_from_spawn_swarm  # 2.0m
        base_y = dist_from_spawn_swarm  # ✅ No Y offset
        base_x = 0.0  # ✅ At origin
        
        wall_positions = [
            (base_x - x_offset, base_y, base_height),
            (base_x + x_offset, base_y, base_height),
            (base_x, base_y + 0.5 * y_offset, base_height),
            (base_x - x_offset, base_y + y_offset, base_height),
            (base_x + x_offset, base_y + y_offset, base_height),
            (base_x, base_y + 1.5 * y_offset, base_height),
            (base_x - x_offset, base_y + 2 * y_offset, base_height),
            (base_x + x_offset, base_y + 2 * y_offset, base_height),
        ]
        
        stage5_start_idx = self.num_drones * 3
        
        for local_idx, (wall_x, wall_y, wall_z) in enumerate(wall_positions):
            global_wall_idx = stage5_start_idx + local_idx
            
            wall_path = f"/World/envs/env_{source_env_idx}/obstacles/wall_{global_wall_idx}"
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
        
        # Collect obstacle positions
        self._collect_obstacle_positions_stage5()

    def _set_stage1_positions(self, env_ids, env_origins):
        """Set hover goals - simplified grid version."""
        ##This function places the drones in a grid formation at the start of the episode and assigns each drone a goal position directly above its start position at a certain height.
        num_reset_envs = len(env_ids)
        
        # Create grid of start positions
        grid_size = int(torch.ceil(torch.sqrt(torch.tensor(self.num_drones, dtype=torch.float32))))
        spacing = torch.zeros(1,device=self.device).uniform_(self.cfg.curriculum.spawn_grid_spacing_range[0], self.cfg.curriculum.spawn_grid_spacing_range[1])#max(0.5, self.cfg.min_safe_distance)
        
        for env_idx in range(num_reset_envs):
            # Random permutation for this environment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            # Sample heights
            start_heights = torch.zeros(self.num_drones, device=self.device).uniform_(0.3, 0.5)
            min_height = self.cfg.curriculum.goal_height_range[0]
            max_height = self.cfg.curriculum.goal_height_range[1]
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
                
                    # Start position
                start_x = env_origins[env_idx, 0] + grid_x
                start_y = env_origins[env_idx, 1] + grid_y
                start_z = start_heights[j]

                # ✅ DEBUG: Print drone positions
                if env_idx == 0:  # Only print for first environment
                    print(f"  - Drone {j}: pos=({start_x.item():.2f}, {start_y.item():.2f}, {start_z.item():.2f}), "
                        f"goal=({start_x.item():.2f}, {start_y.item():.2f}, {goal_heights[j].item():.2f})")
                


                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # Goal position (minimal XY drift)
                xy_noise = torch.zeros(2, device=self.device).uniform_(-0.05, 0.05)
                self._desired_pos_w[env_id_single, j, 0] = default_root_state[0, 0] + xy_noise[0]
                self._desired_pos_w[env_id_single, j, 1] = default_root_state[0, 1] + xy_noise[1]
                self._desired_pos_w[env_id_single, j, 2] = goal_heights[j]
    
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
        spacing = torch.zeros(1,device=self.device).uniform_(self.cfg.curriculum.spawn_grid_spacing_range[0], self.cfg.curriculum.spawn_grid_spacing_range[1])
        
        # ✅ Use configured height range
        min_height = self.cfg.curriculum.goal_height_range[0]
        max_height = self.cfg.curriculum.goal_height_range[1]
        
        # Calculate height stratification to prevent collisions
        # Each drone operates on a different Z-plane
        z_spacing = self.cfg.curriculum.stage2_zdist_xy_plane
        base_height = min_height  # Start from minimum height
        
        for env_idx in range(num_reset_envs):
            # Random permutation for grid assignment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            # Assign heights to create vertical separation
            # Heights increase with drone index to create layered formation
            heights = torch.arange(self.num_drones, device=self.device, dtype=torch.float32)
            heights = base_height + heights * z_spacing
            
            # ✅ Clamp heights to stay within configured range
            heights = torch.clamp(heights, min=min_height, max=max_height)
            
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
                start_z = assigned_heights[j].item()
                
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
                
                # ✅ FIXED: Goal height with variation, clamped to configured range
                # Goal Z stays close to start Z with small variation
                goal_z_noise = torch.zeros(1, device=self.device).uniform_(-0.3, 0.3).item()
                goal_z = torch.clamp(
                    torch.tensor(start_z + goal_z_noise, device=self.device),
                    min=min_height,  # Never below minimum safe height
                    max=max_height   # Never above maximum height
                ).item()
                
                # Set goal
                self._desired_pos_w[env_id_single, j, 0] = goal_x
                self._desired_pos_w[env_id_single, j, 1] = goal_y
                self._desired_pos_w[env_id_single, j, 2] = goal_z
 
    def _set_stage3_positions(self, env_ids, env_origins):
        """Set individual obstacle course navigation with waypoint-based goals."""
        num_reset_envs = len(env_ids)
        offset_x, offset_y = (0, 0)
        
        # ✅ GET SHARED PARAMETERS from config
        params = self.cfg.curriculum.get_stage3_params()
        
        # Height range
        min_height = self.cfg.curriculum.goal_height_range[0]
        max_height = self.cfg.curriculum.goal_height_range[1]
        
        # Agent spawn parameters
        spawn_x = offset_x - 0.6  # ✅ Start 0.5m BEFORE first obstacle
        
        for env_idx in range(num_reset_envs):
            env_id_int = env_ids[env_idx].item()
            env_id_single = env_ids[env_idx].unsqueeze(0)
            
            perm = torch.randperm(self.num_drones, device=self.device)
            base_height = torch.zeros(1, device=self.device).uniform_(min_height, max_height).item()
            
            for j, rob in enumerate(self._robots):
                # -----------------------------------------------------
                # AGENT START POSITION
                # -----------------------------------------------------
                agent_lane = perm[j].item()
                lane_center_y = offset_y + (agent_lane - (self.num_drones - 1) / 2.0) * params["lane_width"]
                
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
                # GENERATE WAYPOINT PATH
                # -----------------------------------------------------
                for wp_idx in range(3):
                    # Obstacle center X position
                    obs_x = env_origins[env_idx, 0] + offset_x + params["course_start_x"] + wp_idx * params["obstacle_spacing_x"]
                    
                    # Zig-zag Y pattern
                    if wp_idx == 0:
                        obs_y = env_origins[env_idx, 1] + lane_center_y
                    elif wp_idx == 1:
                        obs_y = env_origins[env_idx, 1] + lane_center_y - params["lateral_offset"]
                    else:
                        obs_y = env_origins[env_idx, 1] + lane_center_y + params["lateral_offset"]
                    
                    # Waypoint BEHIND obstacle
                    waypoint_x = obs_x + params["waypoint_distance_behind"]
                    waypoint_y = obs_y
                    
                    z_variation = torch.zeros(1, device=self.device).uniform_(-0.2, 0.2).item()
                    waypoint_z = torch.clamp(
                        torch.tensor(base_height + z_variation, device=self.device),
                        min=min_height,
                        max=max_height
                    ).item()
                    
                    # Store waypoint
                    self._waypoint_paths[env_id_int, j, wp_idx, 0] = waypoint_x
                    self._waypoint_paths[env_id_int, j, wp_idx, 1] = waypoint_y
                    self._waypoint_paths[env_id_int, j, wp_idx, 2] = waypoint_z
                
                # Reset waypoint index
                self._current_waypoint_idx[env_id_int, j] = 0
                
                # Set initial goal
                self._desired_pos_w[env_id_single, j, :] = self._waypoint_paths[env_id_int, j, 0, :]
        
        #print(f"[INFO] Stage 3 reset complete using shared parameters")
       
    def _set_stage4_positions(self, env_ids, env_origins):
        """Set swarm navigation goals with formation aligned to travel direction."""
        num_reset_envs = len(env_ids)
        
        spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
            self.cfg.curriculum.goal_height_range[0], 
            self.cfg.curriculum.goal_height_range[1]
        )
        #min_goal_height = self.cfg.curriculum.goal_height_range[0]
        #max_goal_height = self.cfg.curriculum.goal_height_range[1]
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
                self.cfg.curriculum.swarm_translation_distance_range[0],
                self.cfg.curriculum.swarm_translation_distance_range[1]
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
            swarm_goal_height = spawn_heights[env_idx] + torch.zeros(1, device=self.device).uniform_(-0.5, 0.5).item()
            for j in range(self.num_drones):
                start_pos = formation_positions[env_idx, j]
                relative_pos = start_pos[:2] - formation_center[:2]
                
                # Rotate relative position
                rotated_x = cos_theta * relative_pos[0] - sin_theta * relative_pos[1]
                rotated_y = sin_theta * relative_pos[0] + cos_theta * relative_pos[1]
                
                # Translate to goal
                goal_x = formation_center[0] + translation_x + rotated_x
                goal_y = formation_center[1] + translation_y + rotated_y
                goal_z = start_pos[2] + swarm_goal_height
                
                self._desired_pos_w[env_id_single, j, 0] = goal_x
                self._desired_pos_w[env_id_single, j, 1] = goal_y
                self._desired_pos_w[env_id_single, j, 2] = goal_z

    def _set_stage5_positions(self, env_ids, env_origins):
        """Set swarm waypoint navigation through stacked X obstacle pattern.
        
        Swarm navigates through 3 waypoints placed at the centers of the X pattern gaps:
        - Waypoint 1: Center of bottom X (between wall1, wall2, wall3)
        - Waypoint 2: Center of middle gap (between wall4, wall5, wall6)
        - Waypoint 3: Center of top X (between wall6, wall7, wall8)
        
        The formation is rotated 90° to face the +Y direction (toward obstacles).
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of environments, shape (num_reset_envs, 3)
        """
        num_reset_envs = len(env_ids)
        offset_x, offset_y = (0,0)
        
        # Get obstacle configuration
        x_offset = self.cfg.curriculum.stage5_obsx_offset
        y_offset = self.cfg.curriculum.stage5_obsy_offset
        
        # Sample spawn heights
        spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
            self.cfg.curriculum.goal_height_range[0], 
            self.cfg.curriculum.goal_height_range[1]
        )
        
        # -----------------------------------------------------
        # 1. START POSITIONS: Inverted V formation ROTATED to face +Y
        # -----------------------------------------------------
        offset_origins = env_origins.clone()
        offset_origins[:, 0] += offset_x
        offset_origins[:, 1] += offset_y
        
        # Get default formation (apex points in -X direction)
        formation_positions = self.get_inverted_v_formation(env_ids, offset_origins, spawn_heights)
        
        # Rotate formation 90° counterclockwise to face +Y direction
        # Rotation matrix for 90° CCW: [0, -1; 1, 0]
        for env_idx in range(num_reset_envs):
            formation_center = formation_positions[env_idx].mean(dim=0)  # (3,)
            
            for j in range(self.num_drones):
                # Get relative position from formation center
                relative_pos = formation_positions[env_idx, j, :2] - formation_center[:2]  # (2,)
                
                # Apply 90° CCW rotation
                rotated_x = -relative_pos[1]  # New X = -Old Y
                rotated_y = -relative_pos[0]   # New Y = Old X
                
                # Update position
                formation_positions[env_idx, j, 0] = formation_center[0] + rotated_x
                formation_positions[env_idx, j, 1] = formation_center[1] + rotated_y
        
        # Reset robots with rotated formation positions
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
        min_goal_height = self.cfg.curriculum.goal_height_range[0]
        max_goal_height = self.cfg.curriculum.goal_height_range[1]
        # Obstacle base Y position (where obstacles actually start)
        dist_y_from_spawn_swarm =  self.cfg.curriculum.dist_from_spawn_swarm
    
        for env_idx in range(num_reset_envs):
            env_id_int = env_ids[env_idx].item()
            
            base_height = spawn_heights[env_idx].item()
            swarm_goal_height= torch.zeros(1, device=self.device).uniform_(-0.5, 0.5).item()
            
            # Waypoint 1: Gap in bottom X (between wall3 and wall4/5)
            wp1_x = env_origins[env_idx, 0] + offset_x
            wp1_y = env_origins[env_idx, 1] + offset_y +dist_y_from_spawn_swarm +0.75 * y_offset  # Between center and middle
            wp1_z = base_height + swarm_goal_height
            
            self._swarm_waypoint_paths[env_id_int, 0, 0] = wp1_x
            self._swarm_waypoint_paths[env_id_int, 0, 1] = wp1_y
            self._swarm_waypoint_paths[env_id_int, 0, 2] = wp1_z
            
            # Waypoint 2: Gap in middle (between wall4/5 and wall6)
            wp2_x = env_origins[env_idx, 0] + offset_x
            wp2_y = env_origins[env_idx, 1] + offset_y +dist_y_from_spawn_swarm +1.25 * y_offset  # Between middle and center top
            wp2_z = base_height + swarm_goal_height
            
            self._swarm_waypoint_paths[env_id_int, 1, 0] = wp2_x
            self._swarm_waypoint_paths[env_id_int, 1, 1] = wp2_y
            self._swarm_waypoint_paths[env_id_int, 1, 2] = wp2_z
            
            # Waypoint 3: Final position beyond top X
            wp3_x = env_origins[env_idx, 0] + offset_x
            wp3_y = env_origins[env_idx, 1] + offset_y +dist_y_from_spawn_swarm +2.5 * y_offset  # Beyond wall7/8
            wp3_z = base_height + swarm_goal_height
            
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
            
            # Calculate formation center (after rotation)
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
        
        #print(f"[Stage 5] Initialized swarm obstacle navigation with {self.num_swarm_waypoints} waypoints")
        #print(f"[Stage 5] Formation rotated 90° to face +Y direction (toward obstacles)")


###---- Swarm formation methods ----###
    def _compute_swarm_centroid(self):
            """Compute swarm centroid position for each environment (Stages 4 & 5).
            
            Calculates the mean position of all drones in each environment.
            Updates the _swarm_centroid buffer: (num_envs, 3)
            """
            # Stack all drone positions: (num_envs, num_drones, 3)
            swarm_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=1)
            
            # Compute centroid as mean across all drones: (num_envs, 3)
            self._swarm_centroid = swarm_positions.mean(dim=1)

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
        v_angle_rad = torch.deg2rad(torch.tensor(self.cfg.swarm_cfg.formation_v_angle_deg, device=self.device))
        base_sep = self.cfg.swarm_cfg.formation_base_separation
        
        # Scale separation based on max_num_agents to ensure formation fits
        scale_factor = max(1.0, self.cfg.swarm_cfg.min_safe_distance / base_sep)
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
            formation_template[apex_idx, 0] = self.cfg.swarm_cfg.formation_apex_offset  # X position (front)
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
                formation_template[wing_idx, 0] = self.cfg.swarm_cfg.formation_apex_offset + x_offset
                formation_template[wing_idx, 1] = y_offset
            
            # Right wing (positive Y)
            for i in range(right_wing_count):
                wing_idx = left_wing_count + i + 1
                x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)  # Backward
                y_offset = (i + 1) * effective_sep * torch.sin(v_angle_rad)  # Right
                formation_template[wing_idx, 0] = self.cfg.swarm_cfg.formation_apex_offset + x_offset
                formation_template[wing_idx, 1] = y_offset
    
        # Verify minimum separation constraint
        if self.num_drones > 1:
            dists = torch.cdist(formation_template.unsqueeze(0), formation_template.unsqueeze(0)).squeeze(0)
            # Set diagonal to large value to ignore self-distances
            dists = dists + torch.eye(self.num_drones, device=self.device) * 1000.0
            min_dist = dists.min()
            
            # If constraint violated, scale up the formation
            if min_dist < self.cfg.swarm_cfg.min_safe_distance:
                scale_up = self.cfg.swarm_cfg.min_safe_distance / min_dist
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
    
