# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate a checkpoint of an RL agent from skrl with metrics collection.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for evaluation.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import random
import time
import torch
import numpy as np

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import UavSwarm.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


class EvaluationMetricsCollector:
    """Collects and manages evaluation metrics for MARL swarm tasks."""
    
    def __init__(self, num_agents: int, num_episodes: int):
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.episodes_data = []
        
        # Current episode tracking
        self.reset_episode()
    
    def reset_episode(self):
        """Reset metrics for a new episode."""
        self.current_episode = {
            'steps': 0,
            'RM_state_count': {},
            'collision_obs': 0,
            'collision_inter': 0,
            'linear_velocities': [],
            'angular_velocities': [],
            'inter_agent_distances': [],
            'final_distances_to_goal': []
        }
    
    def update_step_metrics(self, obs: dict, env_unwrapped):
        """Update metrics at each step.
        
        Args:
            obs: Observations dictionary from environment
            env_unwrapped: Unwrapped environment to access internal state
        """
        self.current_episode['steps'] += 1
        
        # ✅ ACCESS ROBOT DATA DIRECTLY (not from observations)
        if hasattr(env_unwrapped, '_robots') and len(env_unwrapped._robots) > 0:
            # Extract velocities from robot data (body frame)
            # Stack all robots: (num_drones, num_envs, 3)
            all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in env_unwrapped._robots], dim=0)
            all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in env_unwrapped._robots], dim=0)
            all_positions = torch.stack([rob.data.root_pos_w for rob in env_unwrapped._robots], dim=0)
            
            # Compute magnitudes per agent: (num_drones, num_envs)
            lin_vel_mags = torch.norm(all_lin_vels, dim=2)
            ang_vel_mags = torch.norm(all_ang_vels, dim=2)
            
            # Take first environment (evaluation uses single env)
            lin_vel_mean = lin_vel_mags[:, 0].mean().item()
            ang_vel_mean = ang_vel_mags[:, 0].mean().item()
            
            self.current_episode['linear_velocities'].append(lin_vel_mean)
            self.current_episode['angular_velocities'].append(ang_vel_mean)
            
            # Compute inter-agent distances (first environment only)
            positions_env0 = all_positions[:, 0, :]  # (num_drones, 3)
            distances = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    dist = torch.norm(positions_env0[i] - positions_env0[j]).item()
                    distances.append(dist)
            
            if distances:
                self.current_episode['inter_agent_distances'].append(np.mean(distances))
        
        # ✅ COUNT RM STATES (if available)
        if hasattr(env_unwrapped, '_rm_states'):
            # _rm_states: (num_envs, num_drones)
            rm_states = env_unwrapped._rm_states[0]  # First environment: (num_drones,)
            
            for state_idx in rm_states:
                state_name = self._get_rm_state_name(state_idx.item())
                self.current_episode['RM_state_count'][state_name] = \
                    self.current_episode['RM_state_count'].get(state_name, 0) + 1
        
        # ✅ COUNT COLLISIONS (check multiple possible attributes)
        # Method 1: Check if environment tracks collisions explicitly
        if hasattr(env_unwrapped, '_termination_reasons'):
            # Use detailed termination tracking if available
            if 'collision' in env_unwrapped._termination_reasons:
                # This is a boolean tensor - count if collision happened this step
                collision_this_step = env_unwrapped._termination_reasons['collision'][0].item()
                if collision_this_step:
                    # Increment counter (we don't know which type, so add to obstacle)
                    self.current_episode['collision_obs'] += 1
        
        # Method 2: Check height-based collisions directly
        elif hasattr(env_unwrapped, '_robots'):
            # Check if any robot out of bounds this step
            for rob in env_unwrapped._robots:
                agent_z = rob.data.root_pos_w[0, 2].item()  # First environment
                
                # Check bounds
                if hasattr(env_unwrapped.cfg, 'reward_cfg'):
                    min_height = env_unwrapped.cfg.reward_cfg.min_flight_height
                    max_height = env_unwrapped.cfg.reward_cfg.max_flight_height
                    
                    if agent_z < min_height or agent_z > max_height:
                        self.current_episode['collision_obs'] += 1
                        break  # Count once per step
        
        # Method 3: Check cached obstacle distances for proximity violations
        if hasattr(env_unwrapped, '_cached_obstacle_dists'):
            obstacle_dists = env_unwrapped._cached_obstacle_dists[:, 0]  # (num_drones,) for first env
            
            # Define collision threshold (e.g., < 0.5m from obstacle)
            collision_threshold = 0.5
            close_to_obstacle = (obstacle_dists < collision_threshold).any().item()
            
            if close_to_obstacle:
                self.current_episode['collision_obs'] += 1
        
        # Method 4: Check inter-agent collisions from cached neighbor distances
        if hasattr(env_unwrapped, '_cached_neighbor_rel_pos_b'):
            neighbor_rel_pos = env_unwrapped._cached_neighbor_rel_pos_b[:, 0, :]  # (num_drones, 3)
            neighbor_dists = torch.norm(neighbor_rel_pos, dim=1)  # (num_drones,)
            
            # Define collision threshold for inter-agent (e.g., < 0.3m)
            inter_collision_threshold = 0.3
            too_close = (neighbor_dists < inter_collision_threshold).any().item()
            
            if too_close:
                self.current_episode['collision_inter'] += 1
    
    def finalize_episode(self, env_unwrapped):
        """Finalize metrics when episode ends.
        
        Args:
            env_unwrapped: Unwrapped environment to access final state
        """
        # Compute episode means
        if self.current_episode['linear_velocities']:
            self.current_episode['mean_lin_vel'] = float(np.mean(self.current_episode['linear_velocities']))
        else:
            self.current_episode['mean_lin_vel'] = 0.0
            
        if self.current_episode['angular_velocities']:
            self.current_episode['mean_ang_vel'] = float(np.mean(self.current_episode['angular_velocities']))
        else:
            self.current_episode['mean_ang_vel'] = 0.0
            
        if self.current_episode['inter_agent_distances']:
            self.current_episode['mean_inter_agent_dist'] = float(np.mean(self.current_episode['inter_agent_distances']))
        else:
            self.current_episode['mean_inter_agent_dist'] = 0.0
        
        # Compute final distances to goal
        if hasattr(env_unwrapped, '_robot') and hasattr(env_unwrapped, 'target_pos'):
            positions = env_unwrapped._robot.data.root_pos_w[0, :, :3]  # First environment
            targets = env_unwrapped.target_pos[0]  # First environment
            distances = torch.norm(positions - targets, dim=1).cpu().numpy().tolist()
            self.current_episode['final_distances_to_goal'] = distances
        
        # Remove temporary lists
        del self.current_episode['linear_velocities']
        del self.current_episode['angular_velocities']
        del self.current_episode['inter_agent_distances']
        
        # Store episode data
        self.episodes_data.append(self.current_episode.copy())
    
    def _get_rm_state_name(self, state_idx: int) -> str:
        """Convert RM state index to name."""
        # Adjust based on your RM states
        state_names = {0: 'H', 1: 'S', 2: 'C', 3: 'O'}
        return state_names.get(state_idx, f'State_{state_idx}')
    
    def save_to_json(self, filepath: str):
        """Save collected metrics to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        output_data = {
            'number_episodes': len(self.episodes_data),
            'num_agents': self.num_agents,
            'episodes': self.episodes_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[INFO] Evaluation metrics saved to: {filepath}")
        self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics."""
        if not self.episodes_data:
            return
        
        total_steps = sum(ep['steps'] for ep in self.episodes_data)
        mean_steps = total_steps / len(self.episodes_data)
        
        # Success rate (assuming success if all agents < 0.5m from goal)
        successes = sum(1 for ep in self.episodes_data 
                       if all(d < 0.5 for d in ep['final_distances_to_goal']))
        success_rate = successes / len(self.episodes_data) * 100
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Episodes: {len(self.episodes_data)}")
        print(f"Mean Steps per Episode: {mean_steps:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Mean Linear Velocity: {np.mean([ep['mean_lin_vel'] for ep in self.episodes_data]):.3f} m/s")
        print(f"Mean Angular Velocity: {np.mean([ep['mean_ang_vel'] for ep in self.episodes_data]):.3f} rad/s")
        print(f"Mean Inter-Agent Distance: {np.mean([ep['mean_inter_agent_dist'] for ep in self.episodes_data]):.3f} m")
        print(f"Total Obstacle Collisions: {sum(ep['collision_obs'] for ep in self.episodes_data)}")
        print(f"Total Inter-Agent Collisions: {sum(ep['collision_inter'] for ep in self.episodes_data)}")
        print("="*50 + "\n")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Evaluate skrl agent with metrics collection."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = 1  # Force single environment for evaluation
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "evaluation"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Initialize metrics collector
    num_agents = env_cfg.num_agents if hasattr(env_cfg, 'num_agents') else 5
    metrics_collector = EvaluationMetricsCollector(num_agents, args_cli.num_episodes)

    # Evaluation loop
    print(f"\n[INFO] Starting evaluation for {args_cli.num_episodes} episodes...")
    episode_count = 0
    
    obs, _ = env.reset()
    
    while simulation_app.is_running() and episode_count < args_cli.num_episodes:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            
            # get actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            
            # env stepping
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Update step metrics
            metrics_collector.update_step_metrics(obs, env.unwrapped)
            
            # Check if episode ended
            done = any(terminated.values()) if isinstance(terminated, dict) else terminated.any()
            timeout = any(truncated.values()) if isinstance(truncated, dict) else truncated.any()
            #TODO : verify done condition for multi-agent envs the episode is not terminating when the drones falls on ground
            if done or timeout:
                metrics_collector.finalize_episode(env.unwrapped)
                episode_count += 1
                print(f"[INFO] Episode {episode_count}/{args_cli.num_episodes} completed")
                
                # Reset for next episode
                obs, _ = env.reset()
                metrics_collector.reset_episode()

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Save metrics to JSON
    metrics_filename = f"evaluation_metrics_{task_name}_episodes{args_cli.num_episodes}.json"
    metrics_filepath = os.path.join(log_dir, metrics_filename)
    metrics_collector.save_to_json(metrics_filepath)

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()