# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
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
import logging
import os
import random
from datetime import datetime

import skrl
from packaging import version
import torch
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import UavSwarm.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def inspect_checkpoint(agent, checkpoint_path):
    """Inspect checkpoint for NaN values and print statistics."""
    print("\n" + "="*80)
    print("CHECKPOINT INSPECTION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    
    # ✅ Handle different agent types (PPO, IPPO, MAPPO)
    policy_has_nan = False
    value_has_nan = False
    
    # Get policy network (depends on agent type)
    if hasattr(agent, 'policy'):
        # Single-agent PPO
        policy_nets = {'main': agent.policy}
        value_nets = {'main': agent.value} if hasattr(agent, 'value') else {}
    elif hasattr(agent, 'policies'):
        # MAPPO (uses policies/values dicts per agent)
        policy_nets = agent.policies
        value_nets = agent.values if hasattr(agent, 'values') else {}
    elif hasattr(agent, 'models') and 'policy' in agent.models:
        # IPPO (uses models dict)
        policy_nets = {'main': agent.models['policy']}
        value_nets = {'main': agent.models.get('value', None)}
    else:
        print("  ⚠️  Could not find policy network in agent")
        print(f"  Agent type: {type(agent).__name__}")
        return False
    
    # Check policy networks
    print("\n[Policy Networks]")
    for agent_id, policy_net in policy_nets.items():
        print(f"\nAgent: {agent_id}")
        for name, param in policy_net.named_parameters():
            if torch.isnan(param).any():
                print(f"  ⚠️  NaN detected in {name}")
                policy_has_nan = True
            else:
                print(f"  ✅ {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, min={param.min().item():.6f}, max={param.max().item():.6f}")
    
    # Check value networks
    if value_nets:
        print("\n[Value Networks]")
        for agent_id, value_net in value_nets.items():
            if value_net is not None:
                print(f"\nAgent: {agent_id}")
                for name, param in value_net.named_parameters():
                    if torch.isnan(param).any():
                        print(f"  ⚠️  NaN detected in {name}")
                        value_has_nan = True
                    else:
                        print(f"  ✅ {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    # Check optimizer state
    print("\n[Optimizer State]")
    
    # Handle different optimizer structures
    if hasattr(agent, 'optimizer'):
        # Single optimizer
        optimizers = [('main', agent.optimizer)]
    elif hasattr(agent, 'optimizers'):
        # Multiple optimizers (MAPPO)
        if isinstance(agent.optimizers, dict):
            optimizers = list(agent.optimizers.items())
        else:
            optimizers = [('main', agent.optimizers)]
    else:
        optimizers = []
        print("  ⚠️  No optimizer found")
    
    for opt_name, optimizer in optimizers:
        print(f"\n  Optimizer '{opt_name}':")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"    Param group {i}:")
            print(f"      Learning rate: {param_group['lr']}")
            if 'momentum' in param_group:
                print(f"      Momentum: {param_group['momentum']}")
        
        # Check if optimizer has state (momentum, adaptive rates)
        if optimizer.state:
            print(f"    Optimizer has state for {len(optimizer.state)} parameters")
        else:
            print("    ⚠️  Optimizer state is empty (freshly initialized)")
    
    print("\n" + "="*80)
    
    return policy_has_nan or value_has_nan


def add_nan_detection_hooks(agent):
    """Add forward hooks to detect NaN in network outputs."""
    #import torch
    
    def nan_hook(module, input, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output
        
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print(f"\n⚠️  NaN DETECTED in {module.__class__.__name__}")
                    print(f"   Output {i}: {nan_mask.sum().item()} NaN values out of {out.numel()}")
    
    # ✅ Handle different agent types
    if hasattr(agent, 'policy'):
        # Single-agent PPO
        policy_nets = {'main': agent.policy}
        value_nets = {'main': agent.value} if hasattr(agent, 'value') else {}
    elif hasattr(agent, 'policies'):
        # MAPPO (uses policies/values dicts per agent)
        policy_nets = agent.policies
        value_nets = agent.values if hasattr(agent, 'values') else {}
    elif hasattr(agent, 'models') and 'policy' in agent.models:
        # IPPO (uses models dict)
        policy_nets = {'main': agent.models['policy']}
        value_nets = {'main': agent.models.get('value', None)}
    else:
        print(f"  ⚠️  Could not find networks to add hooks (agent type: {type(agent).__name__})")
        return
    
    # Register hooks on policy networks
    hooks_registered = 0
    for agent_id, policy_net in policy_nets.items():
        if policy_net is not None:
            for module in policy_net.modules():
                module.register_forward_hook(nan_hook)
            hooks_registered += 1
    
    # Register hooks on value networks
    for agent_id, value_net in value_nets.items():
        if value_net is not None:
            for module in value_net.modules():
                module.register_forward_hook(nan_hook)
            hooks_registered += 1
    
    print(f"[INFO] NaN detection hooks registered on {hooks_registered} networks")


def reset_agent_for_new_stage(agent, checkpoint_path):
    """
    Reset agent optimizer and policy parameters for curriculum learning stage.
    
    Args:
        agent: The skrl agent (PPO, IPPO, or MAPPO)
        checkpoint_path: Path to the loaded checkpoint (for logging)
    
    Returns:
        bool: True if checkpoint contains NaN values
    """
    import torch
    
    print("\n" + "="*80)
    print("RESETTING AGENT FOR NEW TRAINING STAGE")
    print("="*80)
    
    # Reset optimizer state for fresh training stage
    print("[INFO] Resetting optimizer state...")
    
    # Arguments that should NOT be passed to optimizer constructor
    OPTIMIZER_NON_CONSTRUCTOR_ARGS = {
        'params', 'differentiable', 'initial_lr', 'capturable', 
        'foreach', 'fused', 'maximize'
    }
    
    if hasattr(agent, 'optimizer'):
        # Single optimizer (PPO)
        opt_class = agent.optimizer.__class__
        opt_config = agent.optimizer.param_groups[0].copy()
        params = [p for group in agent.optimizer.param_groups for p in group['params']]
        
        # Remove non-constructor args
        for key in OPTIMIZER_NON_CONSTRUCTOR_ARGS:
            opt_config.pop(key, None)
        
        # Create fresh optimizer
        agent.optimizer = opt_class(params, **opt_config)
        print("  ✅ Reset single optimizer state")
        
    elif hasattr(agent, 'optimizers'):
        # Multiple optimizers (MAPPO/IPPO)
        if isinstance(agent.optimizers, dict):
            for name, optimizer in agent.optimizers.items():
                opt_class = optimizer.__class__
                opt_config = optimizer.param_groups[0].copy()
                params = [p for group in optimizer.param_groups for p in group['params']]
                
                # Remove non-constructor args
                for key in OPTIMIZER_NON_CONSTRUCTOR_ARGS:
                    opt_config.pop(key, None)
                
                # Create fresh optimizer
                agent.optimizers[name] = opt_class(params, **opt_config)
                print(f"  ✅ Reset optimizer '{name}' state")
        else:
            opt_class = agent.optimizers.__class__
            opt_config = agent.optimizers.param_groups[0].copy()
            params = [p for group in agent.optimizers.param_groups for p in group['params']]
            
            for key in OPTIMIZER_NON_CONSTRUCTOR_ARGS:
                opt_config.pop(key, None)
            
            agent.optimizers = opt_class(params, **opt_config)
            print("  ✅ Reset optimizer state")
    
    # Reset log_std to initial value for exploration
    # Handle different agent structures
    if hasattr(agent, 'policies'):
        # MAPPO (per-agent policies)
        for agent_id, policy in agent.policies.items():
            if hasattr(policy, 'log_std'):
                policy.log_std.data.fill_(0.0)
        print("  ✅ Reset policy log_std to 0.0 for all agents")
    elif hasattr(agent, 'models') and 'policy' in agent.models:
        # IPPO
        agent.models["policy"].log_std.data.fill_(0.0)
        print("  ✅ Reset policy log_std to 0.0")
    elif hasattr(agent, 'policy') and hasattr(agent.policy, 'log_std'):
        # PPO
        agent.policy.log_std.data.fill_(0.0)
        print("  ✅ Reset policy log_std to 0.0")
    
    # Inspect checkpoint for NaN values
    has_nan = inspect_checkpoint(agent, checkpoint_path)
    
    print("="*80 + "\n")
    
    return has_nan


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    num_agents_correct = env.unwrapped.num_agents
    
    # Debug: Print environment info BEFORE wrapping
    print("\n[DEBUG] Environment info before wrapping:")
    print(f"Environment type: {type(env.unwrapped)}")
    print(f"Is DirectMARLEnv: {isinstance(env.unwrapped, DirectMARLEnv)}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        print(f"Number of agents: {env.unwrapped.num_agents}")
        print(f"Possible agents: {env.unwrapped.possible_agents}")
        # Safely access observation/action spaces
        if hasattr(env.unwrapped, 'observation_space'):
            obs_space = env.unwrapped.observation_space
            if isinstance(obs_space, dict):
                print(f"Observation spaces: {list(obs_space.keys())}")
            else:
                print(f"Observation space type: {type(obs_space)}")
        if hasattr(env.unwrapped, 'action_space'):
            act_space = env.unwrapped.action_space
            if isinstance(act_space, dict):
                print(f"Action spaces: {list(act_space.keys())}")
            else:
                print(f"Action space type: {type(act_space)}")
    print()
    
    # Reset environment to initialize buffers
    print("[INFO] Resetting environment to initialize buffers...")
    obs, info = env.reset()
    print(f"[DEBUG] Initial observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"[DEBUG] Observation keys: {list(obs.keys())}")
        # Check first level keys
        for key in list(obs.keys())[:3]:  # Print first 3 keys as sample
            print(f"  - {key}: shape {obs[key].shape if hasattr(obs[key], 'shape') else type(obs[key])}")
    print()
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        print("[INFO] Converting multi-agent environment to single-agent instance for PPO.")
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    
    # Debug: Verify wrapper preserved multi-agent info
    print("\n[DEBUG] Environment info after SkrlVecEnvWrapper:")
    print(f"Wrapped env type: {type(env)}")
    print(f"Has num_agents: {hasattr(env, 'num_agents')}")
    if hasattr(env, 'num_agents'):
        print(f"Number of agents: {env.num_agents}")
    if hasattr(env, 'possible_agents'):
        print(f"Possible agents: {env.possible_agents}")
    if hasattr(env, 'observation_space'):
        print(f"Observation space type: {type(env.observation_space)}")
        if isinstance(env.observation_space, dict):
            print(f"Observation space keys: {list(env.observation_space.keys())[:3]}")  # First 3 keys
    if hasattr(env, 'action_space'):
        print(f"Action space type: {type(env.action_space)}")
        if isinstance(env.action_space, dict):
            print(f"Action space keys: {list(env.action_space.keys())[:3]}")  # First 3 keys
    print()

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    
    # ✅ ADD NaN DETECTION HOOKS (works for all agent types now)
    print(f"\n[DEBUG] Agent type: {type(runner.agent).__name__}")
    print(f"[DEBUG] Agent attributes: {[attr for attr in dir(runner.agent) if not attr.startswith('_')][:10]}")
    add_nan_detection_hooks(runner.agent)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        
        # Reset agent for new training stage
        has_nan = reset_agent_for_new_stage(runner.agent, resume_path)
        
        if has_nan:
            print("\n❌ ERROR: Checkpoint contains NaN values!")
            env.close()
            simulation_app.close()
            sys.exit(1)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()