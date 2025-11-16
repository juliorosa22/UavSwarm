# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv

import UavSwarm.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Check if it's a MARL environment
    is_marl = isinstance(env.unwrapped, DirectMARLEnv)

    # print info (this is vectorized environment)
    if is_marl:
        print(f"[INFO]: MARL Environment detected")
        print(f"[INFO]: Number of agents: {env.unwrapped.num_agents}")
        print(f"[INFO]: Possible agents: {env.unwrapped.cfg.possible_agents}")
        print(f"[INFO]: Action spaces: {env.unwrapped.cfg.action_spaces}")
        print(f"[INFO]: Observation spaces: {env.unwrapped.cfg.observation_spaces}")
    else:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
    
    # reset environment
    env.reset()
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if is_marl:
                # For MARL: actions must be a dictionary {agent_name: action_tensor}
                # Each action tensor has shape (num_envs, action_dim)
                num_envs = env.unwrapped.num_envs
                action_dim = env.unwrapped.cfg.single_action_space
                
                actions = {
                    agent: torch.zeros(num_envs, action_dim, device=env.unwrapped.device)
                    for agent in env.unwrapped.cfg.possible_agents
                }
            else:
                # For standard RL: actions shape from action_space
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
