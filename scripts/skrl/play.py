# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
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
    # note: certain randomization occur in the environment initialization so we set the seed here
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

    # set the log directory for the environment (works for all environment types)
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
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    
    # ✅ DEBUG: Inspect model weights for NaN/Inf
    print(f"\n{'='*80}")
    print(f"[DEBUG] Model Weight Inspection")
    print(f"{'='*80}")
    
    for agent_name, policy in runner.agent.policies.items():
        print(f"\n  Agent: {agent_name}")
        
        # Check if policy has 'net' (actor network)
        if hasattr(policy, 'net'):
            model = policy.net
            print(f"    Actor network: {type(model).__name__}")
            
            # Inspect all parameters
            total_params = 0
            nan_params = 0
            inf_params = 0
            
            for name, param in model.named_parameters():
                total_params += param.numel()
                if torch.isnan(param).any():
                    nan_params += torch.isnan(param).sum().item()
                    print(f"      ❌ NaN in {name}: {torch.isnan(param).sum().item()} values")
                if torch.isinf(param).any():
                    inf_params += torch.isinf(param).sum().item()
                    print(f"      ❌ Inf in {name}: {torch.isinf(param).sum().item()} values")
            
            print(f"    Total parameters: {total_params:,}")
            print(f"    NaN parameters: {nan_params}")
            print(f"    Inf parameters: {inf_params}")
            
            if nan_params > 0 or inf_params > 0:
                print(f"    ⚠️  MODEL HAS CORRUPTED WEIGHTS!")
        
        else:
            print(f"    Policy structure: {type(policy)}")
            print(f"    Available attributes: {[a for a in dir(policy) if not a.startswith('_')][:10]}")
    
    print(f"{'='*80}\n")
    
    # ✅ TEST: Try forward pass with dummy input
    print(f"[DEBUG] Testing forward pass with dummy observations...")
    try:
        dummy_obs = {
            agent: torch.zeros(1, space.shape[0], device=runner.agent.device)
            for agent, space in runner.agent.observation_spaces.items()
        }
        
        test_outputs = runner.agent.act(dummy_obs, timestep=0, timesteps=0)
        
        print(f"  Forward pass successful!")
        for agent_name in runner.agent.possible_agents:
            test_actions = test_outputs[0][agent_name]
            has_nan = torch.isnan(test_actions).any().item()
            has_inf = torch.isinf(test_actions).any().item()
            print(f"    {agent_name}: NaN={has_nan}, Inf={has_inf}, mean={test_actions.mean().item():.4f}")
            
            if has_nan or has_inf:
                print(f"      ⚠️  DUMMY INPUT ALSO PRODUCES NaN - MODEL IS BROKEN!")
                
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
    
    print(f"{'='*80}\n")
    
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    
    # ✅ DEBUG: Check initial observations
    print(f"\n{'='*80}")
    print(f"[DEBUG] Initial Observations After Reset")
    print(f"{'='*80}")
    if isinstance(obs, dict):
        print(f"Type: dict with {len(obs)} agents")
        for agent_name, agent_obs in obs.items():
            print(f"  {agent_name}:")
            print(f"    Shape: {agent_obs.shape}")
            print(f"    Device: {agent_obs.device}")
            print(f"    Dtype: {agent_obs.dtype}")
            print(f"    Has NaN: {torch.isnan(agent_obs).any().item()}")
            print(f"    Has Inf: {torch.isinf(agent_obs).any().item()}")
            print(f"    Min: {agent_obs.min().item():.4f}, Max: {agent_obs.max().item():.4f}")
            print(f"    Mean: {agent_obs.mean().item():.4f}, Std: {agent_obs.std().item():.4f}")
    else:
        print(f"Type: tensor")
        print(f"  Shape: {obs.shape}")
        print(f"  Device: {obs.device}")
        print(f"  Has NaN: {torch.isnan(obs).any().item()}")
        print(f"  Has Inf: {torch.isinf(obs).any().item()}")
        print(f"  Min: {obs.min().item():.4f}, Max: {obs.max().item():.4f}")
    
    # ✅ DEBUG: Check agent state - FIX attribute access
    print(f"\n[DEBUG] Agent Configuration:")
    print(f"  Agent type: {type(runner.agent).__name__}")
    print(f"  Agent class: {runner.agent.__class__}")
    
    # List all available attributes
    print(f"\n  Available attributes:")
    agent_attrs = [attr for attr in dir(runner.agent) if not attr.startswith('_')]
    for attr in sorted(agent_attrs)[:20]:  # Print first 20 non-private attributes
        print(f"    - {attr}")
    
    # Try different attribute names that might exist
    if hasattr(runner.agent, 'observation_space'):
        print(f"  Observation space: {runner.agent.observation_space}")
    elif hasattr(runner.agent, 'observation_spaces'):
        print(f"  Observation spaces: {runner.agent.observation_spaces}")
    elif hasattr(runner.agent, 'num_observations'):
        print(f"  Num observations: {runner.agent.num_observations}")
    else:
        print(f"  Observation space: NOT FOUND (check available attributes above)")
    
    if hasattr(runner.agent, 'action_space'):
        print(f"  Action space: {runner.agent.action_space}")
    elif hasattr(runner.agent, 'action_spaces'):
        print(f"  Action spaces: {runner.agent.action_spaces}")
    elif hasattr(runner.agent, 'num_actions'):
        print(f"  Num actions: {runner.agent.num_actions}")
    else:
        print(f"  Action space: NOT FOUND")
    
    if hasattr(runner.agent, 'device'):
        print(f"  Device: {runner.agent.device}")
    elif hasattr(runner.agent, '_device'):
        print(f"  Device: {runner.agent._device}")
    
    # Check for policies (MAPPO has separate actor/critic)
    if hasattr(runner.agent, 'policy'):
        print(f"  Has policy: True")
    if hasattr(runner.agent, 'policies'):
        print(f"  Has policies dict: True")
        print(f"    Policy keys: {list(runner.agent.policies.keys())}")
    
    # ✅ DEBUG: Check if agent has normalization layers
    if hasattr(runner.agent, 'observation_mean'):
        print(f"  Observation normalization: ENABLED")
        print(f"    Mean shape: {runner.agent.observation_mean.shape}")
        print(f"    Std shape: {runner.agent.observation_std.shape}")
    elif hasattr(runner.agent, '_observation_mean'):
        print(f"  Observation normalization: ENABLED (private attr)")
        print(f"    Mean shape: {runner.agent._observation_mean.shape}")
        print(f"    Std shape: {runner.agent._observation_std.shape}")
    else:
        print(f"  Observation normalization: DISABLED")
    
    if hasattr(runner.agent, 'value_mean'):
        print(f"  Value normalization: ENABLED")
    elif hasattr(runner.agent, '_value_mean'):
        print(f"  Value normalization: ENABLED (private attr)")
    else:
        print(f"  Value normalization: DISABLED")
    
    print(f"{'='*80}\n")
    
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # ✅ DEBUG: Print obs before agent.act (only first few steps)
            if timestep < 3:
                print(f"\n[Step {timestep}] BEFORE agent.act():")
                if isinstance(obs, dict):
                    for agent_name, agent_obs in obs.items():
                        has_nan = torch.isnan(agent_obs).any().item()
                        has_inf = torch.isinf(agent_obs).any().item()
                        print(f"  {agent_name}: NaN={has_nan}, Inf={has_inf}, mean={agent_obs.mean().item():.4f}")
                        if has_nan or has_inf:
                            print(f"    ERROR: Invalid observations detected!")
                            print(f"    First env: {agent_obs[0, :10]}")
                else:
                    has_nan = torch.isnan(obs).any().item()
                    has_inf = torch.isinf(obs).any().item()
                    print(f"  obs: NaN={has_nan}, Inf={has_inf}, mean={obs.mean().item():.4f}")
            
            # agent stepping
            try:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            except Exception as e:
                print(f"\n[ERROR] Exception in agent.act() at timestep {timestep}:")
                print(f"  {type(e).__name__}: {e}")
                print(f"  Observation state:")
                if isinstance(obs, dict):
                    for k, v in obs.items():
                        print(f"    {k}: shape={v.shape}, NaN={torch.isnan(v).any()}, Inf={torch.isinf(v).any()}")
                simulation_app.close()
                return
            
            # ✅ DEBUG: Print outputs from agent.act (only first few steps)
            if timestep < 3:
                print(f"\n[Step {timestep}] AFTER agent.act():")
                print(f"  Output structure: {type(outputs)}, length={len(outputs)}")
                print(f"  outputs[0] (actions): {type(outputs[0])}")
                print(f"  outputs[-1] (dict): {type(outputs[-1])}")
                
                if isinstance(outputs[0], dict):
                    print(f"    Actions dict keys: {list(outputs[0].keys())}")
                    for agent_name, agent_actions in outputs[0].items():
                        has_nan = torch.isnan(agent_actions).any().item()
                        has_inf = torch.isinf(agent_actions).any().item()
                        print(f"    {agent_name}: shape={agent_actions.shape}, NaN={has_nan}, Inf={has_inf}")
                        if has_nan or has_inf:
                            print(f"      ERROR: Invalid actions from agent!")
                            print(f"      Actions: {agent_actions[0]}")
                else:
                    has_nan = torch.isnan(outputs[0]).any().item()
                    has_inf = torch.isinf(outputs[0]).any().item()
                    print(f"    Actions: shape={outputs[0].shape}, NaN={has_nan}, Inf={has_inf}")
            
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                
                # ✅ DEBUG: Validate actions before stepping
                for agent_name, agent_actions in actions.items():
                    if torch.isnan(agent_actions).any():
                        print(f"\n[ERROR] NaN detected in actions for {agent_name} at timestep {timestep}!")
                        print(f"  outputs[0][{agent_name}]: {outputs[0][agent_name]}")
                        print(f"  outputs[-1][{agent_name}]: {outputs[-1][agent_name]}")
                        print(f"  Observation that caused this: {obs[agent_name][0, :10]}")
                        simulation_app.close()
                        return
                    
                    if torch.isinf(agent_actions).any():
                        print(f"\n[ERROR] Inf detected in actions for {agent_name} at timestep {timestep}!")
                        simulation_app.close()
                        return
                
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
                
                if torch.isnan(actions).any():
                    print(f"\n[ERROR] NaN detected in actions at timestep {timestep}!")
                    print(f"  outputs[0]: {outputs[0]}")
                    print(f"  outputs[-1]: {outputs[-1]}")
                    print(f"  Observation that caused this: {obs[0, :10]}")
                    simulation_app.close()
                    return
            
            # ✅ DEBUG: Print actions before stepping (only first few steps)
            if timestep < 3:
                print(f"\n[Step {timestep}] Actions to environment:")
                if isinstance(actions, dict):
                    for agent_name, agent_actions in actions.items():
                        print(f"  {agent_name}: mean={agent_actions.mean().item():.4f}, "
                              f"min={agent_actions.min().item():.4f}, max={agent_actions.max().item():.4f}")
                else:
                    print(f"  mean={actions.mean().item():.4f}, "
                          f"min={actions.min().item():.4f}, max={actions.max().item():.4f}")
            
            # env stepping
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # ✅ DEBUG: Check observations after step (only first few steps)
            if timestep < 3:
                print(f"\n[Step {timestep}] AFTER env.step():")
                if isinstance(obs, dict):
                    for agent_name, agent_obs in obs.items():
                        has_nan = torch.isnan(agent_obs).any().item()
                        has_inf = torch.isinf(agent_obs).any().item()
                        print(f"  {agent_name}: NaN={has_nan}, Inf={has_inf}")
                        if has_nan or has_inf:
                            print(f"    ERROR: Environment returned invalid observations!")
                            print(f"    First env: {agent_obs[0, :10]}")
                            print(f"    This was caused by actions: {actions[agent_name][0]}")
                            simulation_app.close()
                            return
                else:
                    has_nan = torch.isnan(obs).any().item()
                    has_inf = torch.isinf(obs).any().item()
                    print(f"  obs: NaN={has_nan}, Inf={has_inf}")
                print(f"{'='*80}\n")
            
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

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
