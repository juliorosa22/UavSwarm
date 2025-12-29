def update_curriculum_stage(self):
        step = self.global_step
        c = self.cfg.curriculum
        
        if step <= c.stage1_end:
            if self.curriculum_stage != 1:
                print("Changed to curriculum stage 1: Individual Hover")
            self.curriculum_stage = 1
            self.cfg.episode_length_s = c.stage1_episode_length_s
            
        elif step <= c.stage2_end:
            if self.curriculum_stage != 2:
                print("Changed to curriculum stage 2: Individual Point-to-Point")
            self.curriculum_stage = 2
            self.cfg.episode_length_s = c.stage2_episode_length_s
            
        elif step <= c.stage3_end:
            if self.curriculum_stage != 3:
                print("Changed to curriculum stage 3: Individual Point-to-Point with Obstacles")
            self.curriculum_stage = 3
            self.cfg.episode_length_s = c.stage3_episode_length_s
            
        elif step <= c.stage4_end:
            if self.curriculum_stage != 4:
                print("Changed to curriculum stage 4: Swarm Navigation without Obstacles")
            self.curriculum_stage = 4
            self.cfg.episode_length_s = c.stage4_episode_length_s
        
        elif step <= c.stage5_end:
            if self.curriculum_stage != 5:
                    print("Changed to curriculum stage 5: Swarm Navigation with Obstacles")
            self.curriculum_stage = 5
            self.cfg.episode_length_s = c.stage5_episode_length_s  
        else:
            #print("Resetting curriculum to stage 1 after completion of all stages.")
            self.curriculum_stage = 1
            self.global_step = 0
            self.cfg.episode_length_s = c.stage1_episode_length_s




def _get_rewards_old(self) -> dict[str, torch.Tensor]:
        """RM state-aware reward calculation with behavior-specific guidance.
        
        Strategy:
        1. Compute individual reward for each drone based on its RM state
        2. Use minimum reward across all drones (cooperative worst-case)
        3. Main term: distance penalty (negative, approaches 0 as agent reaches goal)
        4. RM state-specific shaping to guide learning behavior
        
        RM State Behaviors:
        - State 0 (Hovering): Prioritize Z-axis movement to gain altitude
        - State 1 (Single-moving): Prioritize smooth XY movement, avoid abrupt changes
        - State 2 (Coop-moving): Same as State 1 + neighbor coordination
        - State 3 (Obstacle-avoiding): Allow aggressive maneuvers, reduce penalties
        """
        
        # Stack all robot data: (num_drones, num_envs, 3)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        
        # ========================================
        # 1. COMPUTE INDIVIDUAL DRONE REWARDS
        # ========================================
        individual_rewards = torch.zeros(self.num_drones, self.num_envs, device=self.device)
        
        # Get RM states: (num_envs, num_drones) -> transpose -> (num_drones, num_envs)
        rm_states = self._rm_states.transpose(0, 1)
        
        # Goal positions transposed: (num_drones, num_envs, 3)
        desired_transposed = self._desired_pos_w.transpose(0, 1)
        
        for drone_idx in range(self.num_drones):
            # Extract drone-specific data
            drone_pos = all_positions[drone_idx]  # (num_envs, 3)
            drone_lin_vel = all_lin_vels[drone_idx]  # (num_envs, 3)
            drone_ang_vel = all_ang_vels[drone_idx]  # (num_envs, 3)
            drone_goal = desired_transposed[drone_idx]  # (num_envs, 3)
            drone_rm_state = rm_states[drone_idx]  # (num_envs,)
            
            # -----------------------------------------------------
            # 1.1 MAIN TERM: DISTANCE PENALTY (ALL STATES)
            # -----------------------------------------------------
            distance = torch.linalg.norm(drone_goal - drone_pos, dim=1)  # (num_envs,)
            distance_penalty = -distance * 0.5  # ✅ REDUCED from 1.0 for better balance
            
            # -----------------------------------------------------
            # 1.2 RM STATE-SPECIFIC SHAPING REWARDS
            # -----------------------------------------------------
            # Create state masks
            is_hovering = (drone_rm_state == 0)  # (num_envs,)
            is_single_moving = (drone_rm_state == 1)
            is_coop_moving = (drone_rm_state == 2)
            is_avoiding = (drone_rm_state == 3)
            
            # ✅ STATE 0 (HOVERING): Prioritize Z-axis movement to gain altitude
            # Reward vertical progress, penalize horizontal drift
            z_error = torch.abs(drone_goal[:, 2] - drone_pos[:, 2])  # Vertical distance
            xy_drift = torch.linalg.norm(drone_goal[:, :2] - drone_pos[:, :2], dim=1)  # Horizontal drift
            
            # Reward for reducing Z error (encourage altitude gain)
            z_progress_reward = -z_error * 0.3 * is_hovering.float()
            
            # Penalty for XY drift when hovering (should stay in place horizontally)
            xy_drift_penalty = -xy_drift * 0.2 * is_hovering.float()
            
            # Heavy penalty for excessive velocity when hovering (should be still)
            hover_vel_penalty = -torch.sum(drone_lin_vel ** 2, dim=1) * 0.03 * is_hovering.float()
            
            # ✅ STATE 1 (SINGLE-MOVING): Smooth movement, avoid abrupt changes
            # Penalize high accelerations (sudden velocity changes)
            # Compute velocity magnitude
            vel_magnitude = torch.linalg.norm(drone_lin_vel, dim=1)  # (num_envs,)
            
            # Penalize excessive velocity (encourage smooth flight)
            smooth_vel_penalty = -torch.clamp(vel_magnitude - 1.0, min=0.0) * 0.1 * is_single_moving.float()
            
            # Penalize angular velocity (avoid spinning/oscillation)
            smooth_ang_penalty = -torch.sum(drone_ang_vel ** 2, dim=1) * 0.01 * is_single_moving.float()
            
            # ✅ STATE 2 (COOP-MOVING): Same as State 1 + neighbor coordination
            # Reuse smooth movement penalties
            coop_vel_penalty = -torch.clamp(vel_magnitude - 1.0, min=0.0) * 0.1 * is_coop_moving.float()
            coop_ang_penalty = -torch.sum(drone_ang_vel ** 2, dim=1) * 0.01 * is_coop_moving.float()
            
            # Additional neighbor coordination reward (only in stages 4-5)
            neighbor_bonus = torch.zeros(self.num_envs, device=self.device)
            if self.curriculum_stage in [4, 5]:
                # Get neighbor data (vectorized version recommended for efficiency)
                # For now, using placeholder - replace with actual neighbor distance computation
                # This should reward maintaining optimal distance to nearest neighbor
                
                # Placeholder: reward proximity to ideal neighbor distance
                # ideal_neighbor_dist = self.cfg.swarm_cfg.max_neighbor_distance / 2.0
                # actual_neighbor_dist = ... (compute from all_positions)
                # neighbor_error = torch.abs(actual_neighbor_dist - ideal_neighbor_dist)
                # neighbor_bonus = -neighbor_error * 0.05 * is_coop_moving.float()
                pass  # ✅ Implement if needed based on your neighbor distance helper
            
            # ✅ STATE 3 (OBSTACLE-AVOIDING): Allow aggressive maneuvers
            # Minimal penalties to allow complex trajectories
            avoid_vel_penalty = -torch.sum(drone_lin_vel ** 2, dim=1) * 0.001 * is_avoiding.float()
            avoid_ang_penalty = -torch.sum(drone_ang_vel ** 2, dim=1) * 0.0005 * is_avoiding.float()
            
            # Bonus for maintaining safe distance from obstacles
            obstacle_clearance_bonus = torch.zeros(self.num_envs, device=self.device)
            if self.curriculum_stage in [3, 5]:
                # Get nearest obstacle distance
                obstacle_dist = self._get_nearest_obstacle_distance_vectorized(
                    all_positions
                )[drone_idx]  # (num_envs,)
                
                # Reward staying above minimum safe distance (1.0m)
                safe_threshold = 1.0
                clearance_ratio = torch.clamp(obstacle_dist / safe_threshold, 0.0, 1.0)
                obstacle_clearance_bonus = clearance_ratio * 0.1 * is_avoiding.float()
            
            # -----------------------------------------------------
            # 1.3 COLLISION PENALTY (ALL STATES)
            # -----------------------------------------------------
            drone_z = drone_pos[:, 2]  # (num_envs,)
            
            too_low = drone_z < self.cfg.reward_cfg.min_flight_height
            too_high = drone_z > self.cfg.reward_cfg.max_flight_height
            collision_penalty = -(too_low | too_high).float() * 5.0  # ✅ REDUCED from 10.0
            
            # -----------------------------------------------------
            # 1.4 COMBINE INDIVIDUAL REWARD
            # -----------------------------------------------------
            individual_rewards[drone_idx] = (
                # MAIN TERM (applies to all states)
                distance_penalty +              # -0.5 * distance (main objective)
                
                # STATE 0 (HOVERING): Z-axis focus
                z_progress_reward +             # Encourage altitude gain
                xy_drift_penalty +              # Discourage horizontal drift
                hover_vel_penalty +             # Encourage stillness
                
                # STATE 1 (SINGLE-MOVING): Smooth movement
                smooth_vel_penalty +            # Avoid high velocities
                smooth_ang_penalty +            # Avoid spinning
                
                # STATE 2 (COOP-MOVING): Smooth + coordination
                coop_vel_penalty +              # Same as State 1
                coop_ang_penalty +              # Same as State 1
                neighbor_bonus +                # Neighbor coordination (stages 4-5)
                
                # STATE 3 (OBSTACLE-AVOIDING): Minimal constraints
                avoid_vel_penalty +             # Minimal velocity penalty
                avoid_ang_penalty +             # Minimal angular penalty
                obstacle_clearance_bonus +      # Reward safe distance
                
                # SAFETY (all states)
                collision_penalty               # Avoid ground/ceiling
            )
        
        # ========================================
        # 2. SWARM-LEVEL PENALTIES (STAGES 4-5)
        # ========================================
        swarm_penalty = torch.zeros(self.num_envs, device=self.device)
        
        if self.curriculum_stage in [4, 5]:
            positions = all_positions.transpose(0, 1)  # (num_envs, num_drones, 3)
            dmat = torch.cdist(positions, positions)
            
            # ✅ SMOOTH inter-agent collision gradient (not binary)
            violations = torch.clamp(
                self.cfg.swarm_cfg.min_safe_distance - dmat, 
                min=0.0
            )
            eye_mask = torch.eye(self.num_drones, device=self.device).unsqueeze(0)
            violations = violations * (1.0 - eye_mask)
            collision_penalty_swarm = -violations.mean(dim=(1, 2)) * 2.0  # ✅ REDUCED from 5.0
            
            # Formation penalty
            mean_dist = torch.mean(dmat, dim=(1, 2))
            formation_penalty = -torch.clamp(
                mean_dist - self.cfg.swarm_cfg.max_formation_distance, 
                min=0.0
            ) * 0.2  # ✅ REDUCED from 0.5
            
            swarm_penalty = collision_penalty_swarm + formation_penalty
        
        # ========================================
        # 3. USE MINIMUM REWARD (WORST-CASE OPTIMIZATION)
        # ========================================
        individual_rewards_T = individual_rewards.transpose(0, 1)  # (num_envs, num_drones)
        min_reward = individual_rewards_T.min(dim=1)[0]  # (num_envs,)
        
        # ✅ REMOVED step_dt scaling for consistency
        reward = min_reward + swarm_penalty
        
        # ========================================
        # 4. LOGGING (for debugging)
        # ========================================
        avg_distance = torch.linalg.norm(
            desired_transposed - all_positions, dim=2
        ).mean(dim=0)
        
        self._episode_sums["distance_to_goal"] += avg_distance
        
        # Log average velocity penalties
        avg_lin_vel = torch.sum(all_lin_vels ** 2, dim=2).mean(dim=0)
        avg_ang_vel = torch.sum(all_ang_vels ** 2, dim=2).mean(dim=0)
        
        self._episode_sums["lin_vel"] += avg_lin_vel
        self._episode_sums["ang_vel"] += avg_ang_vel
        
        # Log collision and formation
        agent_z = all_positions[:, :, 2]
        too_low = agent_z < self.cfg.reward_cfg.min_flight_height
        too_high = agent_z > self.cfg.reward_cfg.max_flight_height
        collision_log = -(too_low | too_high).any(dim=0).float() * 5.0
        
        self._episode_sums["collision"] += collision_log
        self._episode_sums["formation"] += swarm_penalty
        
        return {f"robot_{i}": reward for i in range(self.num_drones)}

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