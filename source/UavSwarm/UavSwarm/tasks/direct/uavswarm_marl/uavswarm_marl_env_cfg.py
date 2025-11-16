# ============================================================
# Swarm MARL Environment Configuration (Direct-style, Crazyflie)
# ============================================================

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
class UavswarmMarlEnvCfg(DirectMARLEnvCfg):
    """
    Direct MARL workflow for UAV Swarm with multiple Crazyflies per environment.
    """

    # ----- Episode / stepping -----
    episode_length_s = 10.0
    decimation = 2
    num_agents: int = 3
    max_num_agents: int = 20
    
    # MARL-specific: Define spaces for all agents
    # Per-agent action/observation dimensions
    single_action_space = 4  # thrust + 3 moments per drone
    single_observation_space = 12  # lin_vel[3] + ang_vel[3] + gravity[3] + desired_pos[3]
    
    # Required for DirectMARLEnvCfg - using robot names as keys
    possible_agents = [f"robot_{i}" for i in range(num_agents)]
    action_spaces = {f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(num_agents)}
    observation_spaces = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(12,)) for i in range(num_agents)}
    
    state_space = 0
    debug_vis = True

    ui_window_class_type = UavSwarmEnvWindow

    # ----- Simulation -----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
        debug_vis=False,
    )

    # ----- Scene -----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # ----- Robot Template (will be instantiated N times) -----
    robot_template: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Action -> force/torque conversion
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ----- Reward scales (matching copy_quadenv.py) -----
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    
    # ----- Formation/Collision parameters -----
    formation_penalty_scale = -0.15
    collision_penalty_scale = -2.0
    
    # ----- Formation parameters -----
    v_spacing_xy = (1.0, 0.8)   # (X step, Y spacing for V formation)
    spawn_height = 1.0
    min_sep = 0.6                # minimum separation distance
    max_sep_mean = 6.0           # maximum mean distance
    goal_height = 1.0
    goal_xy_range = 2.0          # goal sampling range