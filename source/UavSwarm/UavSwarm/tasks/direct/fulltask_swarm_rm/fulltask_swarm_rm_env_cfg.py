# ============================================================
# Swarm MARL Environment Configuration (Direct-style, Crazyflie)
# ============================================================
### TODO adjust this env cfg for full task swarm combined with Reward Machines
from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Predefined asset
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import CRAZYFLIE_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass as omniclass


class UavSwarmEnvWindow(BaseEnvWindow):
    """Window manager for the UAV Swarm environment."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class CurriculumCfg:
    active_stage: int = 1  # Current active stage (1 to 5)
    
# ✅ NEW: Episode durations per stage
    stage1_episode_length_s: float = 30.0   # Hover
    stage2_episode_length_s: float = 40.0   # Point-to-point
    stage3_episode_length_s: float = 90.0   # Obstacles
    stage4_episode_length_s: float = 60.0   # Swarm
    stage5_episode_length_s: float = 120.0  # Swarm + obstacles

    ##--- Stage 2 parameters ---##
    stage2_goal_distance:float = 6.0  # xy-distance for point-to-point and obstacle navigation stages
    stage2_zdist_xy_plane: float = 1.0  # z- distance between each agent xy-plane this will mitigate collisions during the training
    
   
    ##--- Stage 3 parameters ---##
    ##--- Stage 3 parameters (✅ NEW: Centralized dictionary) ---##
    stage3_params: dict = {
        "lane_width": 1.2,              # Y-spacing between agent lanes
        "obstacle_spacing_x": 2.5,      # X-distance between consecutive obstacles
        "lateral_offset": 0.3,          # Y-offset for zig-zag pattern
        "course_start_x": 1.5,          # First obstacle X position from origin
        "waypoint_distance_behind": 1.0 # Distance behind obstacle for waypoint placement
    }
    
    ## --- Stage 4 parameters ---##

    swarm_translation_distance_range: tuple = (2, 5)  # Minimum formation translation (meters)
    

    ##--- Stage 5 parameters ---##
    #now built on origin (0,0) with obstacle field centered there
    #stage5_offset_x: float = 0.0   # Keep X centered for stage 5
    #stage5_offset_y: float = 15.0  # Move agents 15m in Y for stage 5
    # Stage 5 obstacle distribution (stacked X pattern)
    stage5_obsx_offset: float = 1.5  # X spacing between wall segments in X pattern
    stage5_obsy_offset: float = 2.0  # Y spacing between wall segments in X pattern
    dist_from_spawn_swarm: float = 2.0  # Distance from spawn to first obstacle in swarm stage
    ##--- Common parameters ---##
    spawn_height_range: tuple = (0.5, 0.8)  # spawn height range for all stages
    spawn_grid_spacing_range: tuple = (0.5, 0.8)  # spacing range for grid spawn positions 
    goal_height_range: tuple = (1.5, 6.0)  # goal height range for all stages
    max_obstacle_distance: float = 10.0  # Maximum distance to clamp obstacle measurements (meters)
    obstacles_size: tuple = (0.15, 0.8, 8.0)  # Default obstacle dimensions (X, Y, Z) in meters

    # ✅ NEW: Method to get episode length for active stage
    def get_episode_length(self) -> float:
        """Return episode length based on active stage.
        
        Returns:
            Episode length in seconds for the current active stage.
        """
        stage_lengths = {
            1: self.stage1_episode_length_s,
            2: self.stage2_episode_length_s,
            3: self.stage3_episode_length_s,
            4: self.stage4_episode_length_s,
            5: self.stage5_episode_length_s,
        }
        
        if self.active_stage not in stage_lengths:
            raise ValueError(f"Invalid active_stage: {self.active_stage}. Must be 1-5.")
        
        return stage_lengths[self.active_stage]
    
    # ✅ NEW: Helper method to get Stage 3 parameters
    def get_stage3_params(self) -> dict:
        return self.stage3_params.copy()

@configclass
class RewardMachineCfg:
    """Configuration for Reward Machine parameters."""

     # ----- Reward Machine States Parameters -----
    hover_min_altitude: float = 0.8  # Minimum altitude to exit hovering state
    close_obs_dist_thresh: float = 0.3  # Distance threshold to enter obstacle-avoiding state
    num_rm_states: int = 4  # Number of RM states

    # ✅ NEW: Termination thresholds
    min_flight_height: float = 0.1  # Minimum safe altitude (meters)
    max_flight_height: float = 15.0  # Maximum safe altitude (meters)
    max_distance_from_origin: float = 15.0  # Maximum XY distance from env origin (meters)
    
    # ✅ NEW: Goal reaching thresholds (stage-dependent)
    hover_position_threshold: float = 0.15  # Stage 1: Position error for hover (meters)
    hover_velocity_threshold: float = 0.2   # Stage 1: Velocity threshold for stable hover (m/s)
    goal_position_threshold: float = 0.15    # Stages 2-3: Position threshold for goals (meters)
    swarm_goal_threshold: float = 0.3      # Stages 4-5: Formation goal threshold (meters)


    # ✅ REBALANCED BASE REWARD SCALES
    lin_vel_reward_scale: float = -0.01  # Reduced from -0.05
    ang_vel_reward_scale: float = -0.01  # Reduced from -0.05
    distance_to_goal_reward_scale: float = 5.0  # Reduced from 1.5
    
    formation_penalty_scale: float = -0.3  # Reduced from -0.5
    collision_penalty_scale: float = -5.0  # Reduced from -10.0
    


    # ✅ NEW: RM STATE-AWARE BONUS SCALES
    altitude_bonus_scale: float = 0.5  # Bonus for maintaining target altitude (Hovering state)
    obstacle_bonus_scale: float = 0.8  # Bonus for safe obstacle clearance (Obstacle-avoiding state)
    neighbor_bonus_scale: float = 0.5  # Bonus for optimal neighbor distance (Coop-moving state)
    state_progress_scale: float = 0.3  # Bonus for state progression (all states)
    
    # ✅ NEW: Safe distance thresholds
    safe_obstacle_distance: float = 0.4  # Meters - distance considered "safe" from obstacles
    optimal_neighbor_distance: float = 0.5  # Meters - target distance between cooperating neighbors

@configclass
class SwarmParameterCfg:
    """Configuration for Swarm parameters."""

    # ----- Swarm Sensing Parameters -----
    max_neighbor_distance: float = 15.0  # Maximum distance for neighbor sensing (meters)
    min_safe_distance: float = 1       # Minimum safe distance between drones (meters)
    optimal_distance: float = 3.0        # Optimal distance for cooperation (meters)
    max_formation_distance: float = 6.0  # Maximum mean distance for formation
    # ----- Inverted V Formation parameters -----
    # Inverted V: apex at front (negative X), wings spread back (positive X) and outward (±Y)
    formation_base_separation = 0.8      # Base separation between adjacent drones
    formation_v_angle_deg = 60.0         # Angle of the V arms (degrees from vertical)
    formation_apex_offset = 0.0          # Forward offset for apex drone
                       # Maximum mean distance (for large formations)


@configclass
class FullTaskUAVSwarmEnvCfg(DirectMARLEnvCfg):
    """
    Direct MARL workflow for UAV Swarm with multiple Crazyflies per environment.
    """

    # ----- Episode / stepping -----
    episode_length_s = 30.0
    decimation = 2
    num_agents: int = 5
    max_num_agents: int = 20

    curriculum:CurriculumCfg = CurriculumCfg()
    reward_cfg: RewardMachineCfg = RewardMachineCfg()
    swarm_cfg: SwarmParameterCfg = SwarmParameterCfg()
    # ----- Obstacle Distance Observation -----
    

    
    #------ENV Spaces dimensions
    # 12 base + 1 obstacle distance + 3 neighbor relative velocity + 3 neighbor relative position + 4 RM state (one-hot) = 23
    single_observation_space = 23  
    # Policy- Action dimensions
    single_action_space = 4  # thrust + 3 moments per drone
    
    #Critic- Observation dimensions
    state_space = num_agents * single_observation_space  
    
    # Required for DirectMARLEnvCfg - using robot names as keys
    possible_agents = [f"robot_{i}" for i in range(num_agents)]
    action_spaces = {f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(num_agents)}
    observation_spaces = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(23,)) for i in range(num_agents)}
    # ----- UI -----
    debug_vis = True
    ui_window_class_type = UavSwarmEnvWindow
    # ----- Simulation -----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, #each step is 10ms
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.2),  # Dark gray (R, G, B)
            roughness=0.8,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    # ----- Scene -----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=10.0, replicate_physics=True, clone_in_fabric=True
    )

    # ----- Robot Template (will be instantiated N times) -----
    robot_template: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Black color (R, G, B)
                roughness=0.5,
                metallic=0.2,
            ),
            visual_material_path="/World/Looks/CrazyflieBlack",  # Unique path for material
        )
    )

    # Action -> force/torque conversion
    thrust_to_weight = 1.9
    moment_scale = 0.01
        
    
    
    