# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Modified to use only robot_0's policy for all agents in multi-agent scenarios.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
parser.add_argument(
    "--eval_agent_id",
    type=str,
    default="robot_0",
    help="Which agent's policy to use for all agents (default: robot_0)",
)

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
import os
import random
import time
import torch

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


def extract_single_policy(agent, eval_agent_id: str):
    """Extract a single agent's policy from the loaded checkpoint."""
    print(f"\n{'='*80}")
    print(f"[INFO] Extracting Single Policy for All Agents")
    print(f"{'='*80}")
    print(f"Using policy from: {eval_agent_id}")
    
    # Try to find the policy in the agent structure
    eval_policy = None
    
    # Method 1: Check for models dict (MAPPO structure)
    if hasattr(agent, 'models') and isinstance(agent.models, dict):
        if eval_agent_id in agent.models:
            agent_models = agent.models[eval_agent_id]
            if isinstance(agent_models, dict) and 'policy' in agent_models:
                eval_policy = agent_models['policy']
                print(f"✅ Found policy in agent.models['{eval_agent_id}']['policy']")
    
    # Method 2: Check for policies dict
    if eval_policy is None and hasattr(agent, 'policies') and isinstance(agent.policies, dict):
        if eval_agent_id in agent.policies:
            eval_policy = agent.policies[eval_agent_id]
            print(f"✅ Found policy in agent.policies['{eval_agent_id}']")
    
    # Method 3: Single policy for all agents
    if eval_policy is None and hasattr(agent, 'policy'):
        eval_policy = agent.policy
        print(f"✅ Using shared policy from agent.policy")
    
    if eval_policy is None:
        raise ValueError(
            f"Could not find policy for '{eval_agent_id}'!\n"
            f"Available attributes: {[a for a in dir(agent) if not a.startswith('_')][:20]}"
        )
    
    # Print policy info
    print(f"\nPolicy Details:")
    print(f"  Type: {type(eval_policy).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in eval_policy.parameters()):,}")
    
    # Check for NaN/Inf in policy weights
    nan_count = 0
    inf_count = 0
    for name, param in eval_policy.named_parameters():
        if torch.isnan(param).any():
            nan_count += torch.isnan(param).sum().item()
            print(f"  ❌ NaN in {name}: {torch.isnan(param).sum().item()} values")
        if torch.isinf(param).any():
            inf_count += torch.isinf(param).sum().item()
            print(f"  ❌ Inf in {name}: {torch.isinf(param).sum().item()} values")
    
    if nan_count == 0 and inf_count == 0:
        print(f"  ✅ All weights are valid")
    else:
        print(f"  ❌ Found {nan_count} NaN and {inf_count} Inf values")
        raise RuntimeError("Policy has corrupted weights!")
    
    print(f"{'='*80}\n")
    
    return eval_policy


def get_actions_from_single_policy(eval_policy, obs_dict, possible_agents, device):
    """Get actions for all agents using a single policy.
    
    Args:
        eval_policy: The policy network to use
        obs_dict: Dictionary of observations {agent_id: obs_tensor}
        possible_agents: List of agent IDs
        device: Torch device
    
    Returns:
        Dictionary of actions {agent_id: action_tensor}
    """
    actions = {}
    
    for agent_id in possible_agents:
        # Prepare input for policy
        policy_input = {"states": obs_dict[agent_id]}
        
        try:
            # Get action from policy
            result = eval_policy.act(policy_input, role="policy")
            
            # Extract action (handle different return formats)
            if isinstance(result, tuple):
                action = result[0]  # First element is always the action
            elif isinstance(result, dict):
                action = result.get("action", result.get("actions", None))
                if action is None:
                    raise ValueError("Cannot find 'action' or 'actions' in dict output")
            elif torch.is_tensor(result):
                action = result
            else:
                raise TypeError(f"Unexpected policy output type: {type(result)}")
            
            # Try to get mean for deterministic actions
            if hasattr(eval_policy, 'distribution'):
                try:
                    dist_result = eval_policy.distribution(policy_input)
                    if isinstance(dist_result, tuple) and len(dist_result) >= 1:
                        mean = dist_result[0]
                        if torch.is_tensor(mean):
                            action = mean  # Use mean instead of sampled action
                except:
                    pass  # Stick with sampled action
            
            actions[agent_id] = action
            
        except Exception as e:
            print(f"  ❌ Error getting action for {agent_id}: {e}")
            # Fallback: zero actions
            action_dim = 4  # Default for UAV task
            actions[agent_id] = torch.zeros(
                obs_dict[agent_id].shape[0],
                action_dim,
                device=device
            )
    
    return actions


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent using single policy for all agents."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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
        print(f"[INFO] Using user-specified checkpoint path: {resume_path}")
    else:
        print(f"[INFO] Searching for latest checkpoint in: {log_root_path}")
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Store whether this is multi-agent BEFORE potential conversion
    is_multi_agent = isinstance(env.unwrapped, DirectMARLEnv)
    
    # convert to single-agent instance if required by the RL algorithm
    if is_multi_agent and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
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
    
    # ✅ Extract single policy for evaluation (only for multi-agent)
    eval_policy = None
    if is_multi_agent and hasattr(env, "possible_agents"):
        eval_policy = extract_single_policy(runner.agent, args_cli.eval_agent_id)
        print(f"[INFO] 🎯 Using ONLY {args_cli.eval_agent_id}'s policy for all {len(env.possible_agents)} agents")
    
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    
    print(f"\n{'='*80}")
    print(f"[INFO] Starting Evaluation")
    print(f"{'='*80}")
    if is_multi_agent and hasattr(env, "possible_agents"):
        print(f"Mode: Multi-agent with SINGLE POLICY")
        print(f"Policy source: {args_cli.eval_agent_id}")
        print(f"Number of agents: {len(env.possible_agents)}")
        print(f"Agent IDs: {env.possible_agents}")
    else:
        print(f"Mode: Single-agent")
    print(f"{'='*80}\n")
    
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # ✅ Use single policy for all agents if multi-agent
            if eval_policy is not None:
                print("Evaluating with single policy for all agents...")
                # Multi-agent mode with single policy
                actions = get_actions_from_single_policy(
                    eval_policy,
                    obs,
                    env.possible_agents,
                    env.device
                )
                
                # Validate actions
                for agent_name, agent_actions in actions.items():
                    if torch.isnan(agent_actions).any():
                        print(f"\n[ERROR] NaN detected in actions for {agent_name} at timestep {timestep}!")
                        print(f"  Observation: {obs[agent_name][0, :10]}")
                        simulation_app.close()
                        return
                    if torch.isinf(agent_actions).any():
                        print(f"\n[ERROR] Inf detected in actions for {agent_name} at timestep {timestep}!")
                        simulation_app.close()
                        return
            
            else:
                # Original behavior: use runner.agent.act()
                try:
                    print("Evaluating with original multi-agent policies...")
                    outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                except Exception as e:
                    print(f"\n[ERROR] Exception in agent.act() at timestep {timestep}:")
                    print(f"  {type(e).__name__}: {e}")
                    simulation_app.close()
                    return
                
                # Extract actions
                if hasattr(env, "possible_agents"):
                    # Multi-agent (deterministic) actions
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    # Single-agent (deterministic) actions
                    actions = outputs[-1].get("mean_actions", outputs[0])
                
                # Validate actions
                if hasattr(env, "possible_agents"):
                    for agent_name, agent_actions in actions.items():
                        if torch.isnan(agent_actions).any() or torch.isinf(agent_actions).any():
                            print(f"\n[ERROR] Invalid actions for {agent_name} at timestep {timestep}!")
                            simulation_app.close()
                            return
                else:
                    if torch.isnan(actions).any() or torch.isinf(actions).any():
                        print(f"\n[ERROR] Invalid actions at timestep {timestep}!")
                        simulation_app.close()
                        return
            
            # env stepping
            obs, rewards, terminated, truncated, info = env.step(actions)
            
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
