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

# ✅ ============================================================================
# ✅ AGENT CONFIGURATION ENTRY POINT SELECTION
# ✅ ============================================================================

# Get algorithm from command line and normalize to lowercase
algorithm = args_cli.algorithm.lower()

# Determine which agent config file to load based on algorithm or explicit --agent arg
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

# import logger
logger = logging.getLogger(__name__)

from UavSwarm.tasks.direct.fulltask_swarm_rm.agents.skrl_shared_model import (
    SharedPolicyModel,
    CentralizedCriticModel,
    AdaptiveCentralizedCriticModel,
)


import UavSwarm.tasks  # noqa: F401

# Add import at the top
import torch
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from gymnasium import spaces


def config_mappo_agent(
    env,
    agent_cfg: dict,
    obs_space,
    act_space,
    state_dim: int,
    num_agents: int,
    models: dict,
) -> tuple[MAPPO, dict]:
    """Configure and create MAPPO agent with custom models.
    
    Args:
        env: The wrapped skrl environment
        agent_cfg: Agent configuration dictionary from YAML
        obs_space: Per-agent observation space (gym.Space)
        act_space: Per-agent action space (gym.Space)
        state_dim: Centralized state dimension (int)
        num_agents: Number of agents in the environment
        models: Dictionary of models {agent_id: {'policy': model, 'value': model}}
    
    Returns:
        tuple: (MAPPO agent instance, processed config dictionary)
    """
    print("\n[INFO] Configuring MAPPO agent...")
    
    # 1. Create memory
    memory_size = agent_cfg["agent"].get("rollouts", 64)
    shared_memory = RandomMemory(
        memory_size=memory_size,
        num_envs=env.num_envs,
        device=env.device
    )
    memories = {agent_id: shared_memory for agent_id in env.possible_agents}
    print(f"[INFO] Created shared memory: size={memory_size}, num_envs={env.num_envs}")
    
    # 2. Prepare base agent config
    mappo_cfg = MAPPO_DEFAULT_CONFIG.copy()
    mappo_cfg.update(agent_cfg.get("agent", {}))
    
    # 3. Configure learning rate scheduler
    mappo_cfg = _configure_scheduler(mappo_cfg)
    
    # 4. Configure preprocessors
    mappo_cfg = _configure_preprocessors(
        mappo_cfg, 
        obs_space, 
        act_space, 
        state_dim, 
        env.device
    )
    
    # 5. Create space dictionaries
    observation_spaces = {agent_id: obs_space for agent_id in env.possible_agents}
    action_spaces = {agent_id: act_space for agent_id in env.possible_agents}
    
    # Create centralized state space object
    state_space_obj = spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=(state_dim,)
    )
    shared_observation_spaces = {agent_id: state_space_obj for agent_id in env.possible_agents}
    
    print(f"[INFO] Created space dicts:")
    print(f"  - observation_spaces: {len(observation_spaces)} agents, shape {obs_space.shape}")
    print(f"  - action_spaces: {len(action_spaces)} agents, shape {act_space.shape}")
    print(f"  - shared_observation_spaces: {len(shared_observation_spaces)} agents, shape {state_space_obj.shape}")
    
    # 6. Instantiate MAPPO agent
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        device=env.device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )
    
    print(f"✅ MAPPO agent created successfully")
    
    return agent, mappo_cfg


def _configure_scheduler(mappo_cfg: dict) -> dict:
    """Convert learning rate scheduler from string to class reference.
    
    Args:
        mappo_cfg: MAPPO configuration dictionary
    
    Returns:
        dict: Updated configuration with scheduler class reference
    """
    if "learning_rate_scheduler" not in mappo_cfg:
        return mappo_cfg
    
    scheduler_name = mappo_cfg["learning_rate_scheduler"]
    if not isinstance(scheduler_name, str):
        # Already a class reference
        return mappo_cfg
    
    # Map scheduler names to classes
    scheduler_map = {
        "KLAdaptiveRL": KLAdaptiveRL,
        "KLAdaptiveLR": KLAdaptiveRL,  # Common typo/alias
    }
    
    if scheduler_name in scheduler_map:
        mappo_cfg["learning_rate_scheduler"] = scheduler_map[scheduler_name]
        print(f"[INFO] Using learning rate scheduler: {scheduler_name}")
    else:
        print(f"[WARNING] Unknown scheduler '{scheduler_name}', using constant learning rate")
        print(f"[WARNING] Available schedulers: {list(scheduler_map.keys())}")
        mappo_cfg.pop("learning_rate_scheduler", None)
        mappo_cfg.pop("learning_rate_scheduler_kwargs", None)
    
    return mappo_cfg


def _configure_preprocessors(
    mappo_cfg: dict,
    obs_space,
    act_space,
    state_dim: int,
    device: torch.device,
) -> dict:
    """Configure preprocessors with proper class references and kwargs.
    
    Args:
        mappo_cfg: MAPPO configuration dictionary
        obs_space: Per-agent observation space
        act_space: Per-agent action space
        state_dim: Centralized state dimension
        device: Torch device
    
    Returns:
        dict: Updated configuration with preprocessor classes and kwargs
    """
    # Map preprocessor names to classes
    preprocessor_map = {
        "RunningStandardScaler": RunningStandardScaler,
    }
    
    # Create centralized state space object
    state_space_obj = spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=(state_dim,)
    )
    
    # Configure each preprocessor type
    preprocessor_configs = [
        {
            "key": "state_preprocessor",
            "kwargs_key": "state_preprocessor_kwargs",
            "size": obs_space,
            "name": "State preprocessor (per-agent observations)"
        },
        {
            "key": "shared_state_preprocessor",
            "kwargs_key": "shared_state_preprocessor_kwargs",
            "size": state_space_obj,
            "name": "Shared state preprocessor (centralized critic)"
        },
        {
            "key": "value_preprocessor",
            "kwargs_key": "value_preprocessor_kwargs",
            "size": 1,  # Value is scalar
            "name": "Value preprocessor (value function output)"
        },
    ]
    
    for config in preprocessor_configs:
        mappo_cfg = _configure_single_preprocessor(
            mappo_cfg,
            config["key"],
            config["kwargs_key"],
            config["size"],
            device,
            preprocessor_map,
            config["name"]
        )
    
    return mappo_cfg


def _configure_single_preprocessor(
    mappo_cfg: dict,
    preprocessor_key: str,
    kwargs_key: str,
    size,
    device: torch.device,
    preprocessor_map: dict,
    description: str,
) -> dict:
    """Configure a single preprocessor.
    
    Args:
        mappo_cfg: MAPPO configuration dictionary
        preprocessor_key: Key for preprocessor class in config
        kwargs_key: Key for preprocessor kwargs in config
        size: Size parameter for preprocessor (gym.Space or int)
        device: Torch device
        preprocessor_map: Mapping of names to preprocessor classes
        description: Human-readable description for logging
    
    Returns:
        dict: Updated configuration
    """
    if preprocessor_key not in mappo_cfg:
        return mappo_cfg
    
    preprocessor_name = mappo_cfg[preprocessor_key]
    if not isinstance(preprocessor_name, str):
        # Already a class reference
        return mappo_cfg
    
    if preprocessor_name not in preprocessor_map:
        print(f"[WARNING] Unknown preprocessor '{preprocessor_name}' for {description}")
        print(f"[WARNING] Available: {list(preprocessor_map.keys())}")
        print(f"[WARNING] Disabling {preprocessor_key}")
        mappo_cfg.pop(preprocessor_key, None)
        mappo_cfg.pop(kwargs_key, None)
        return mappo_cfg
    
    # Convert string to class reference
    mappo_cfg[preprocessor_key] = preprocessor_map[preprocessor_name]
    print(f"[INFO] Using {description}: {preprocessor_name}")
    
    # Ensure kwargs exist and have proper structure
    if kwargs_key not in mappo_cfg or mappo_cfg[kwargs_key] is None:
        mappo_cfg[kwargs_key] = {}
    
    # Add required 'size' and 'device' if not present
    if "size" not in mappo_cfg[kwargs_key]:
        mappo_cfg[kwargs_key]["size"] = size
    if "device" not in mappo_cfg[kwargs_key]:
        mappo_cfg[kwargs_key]["device"] = device
    
    return mappo_cfg


def create_shared_models(
    obs_space,
    act_space,
    state_dim: int,
    num_agents: int,
    possible_agents: list,
    device: torch.device,
) -> dict:
    """Create shared policy and centralized critic models.
    
    Args:
        obs_space: Per-agent observation space
        act_space: Per-agent action space
        state_dim: Centralized state dimension
        num_agents: Number of agents
        possible_agents: List of agent IDs
        device: Torch device
    
    Returns:
        dict: Models dictionary {agent_id: {'policy': model, 'value': model}}
    """
    print("\n[INFO] Creating custom shared policy and centralized critic...")
    print(f"[INFO] Number of agents: {num_agents}")
    print(f"[INFO] Observation space per agent: {obs_space.shape}")
    print(f"[INFO] Action space per agent: {act_space.shape}")
    print(f"[INFO] State space (centralized): {state_dim} dims")
    
    # Create ONE shared policy for all agents
    shared_policy = SharedPolicyModel(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
    )
    
    # Create ONE centralized critic
    state_space_obj = spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=(state_dim,)
    )
    
    centralized_critic = CentralizedCriticModel(
        observation_space=state_space_obj,
        action_space=act_space,
        device=device,
        clip_actions=False,
        num_agents=num_agents,
    )
    
    # Assign SAME model instance to all agents
    models = {}
    for agent_id in possible_agents:
        models[agent_id] = {
            "policy": shared_policy,
            "value": centralized_critic,
        }
    
    print(f"[INFO] ✅ Created shared models for all {num_agents} agents")
    print(f"[INFO] Policy parameters: {sum(p.numel() for p in shared_policy.parameters()):,}")
    print(f"[INFO] Critic parameters: {sum(p.numel() for p in centralized_critic.parameters()):,}")
    
    # Verify model sharing
    print("\n[INFO] Verifying weight sharing...")
    policy_0 = models[possible_agents[0]]["policy"]
    all_same = all(
        models[agent_id]["policy"] is policy_0
        for agent_id in possible_agents[1:]
    )
    if all_same:
        print(f"✅ Confirmed: All {num_agents} agents share the SAME policy network")
    else:
        print(f"⚠️  Warning: Agents have SEPARATE policy networks")
    
    return models


def load_checkpoint_with_validation(
    agent: MAPPO,
    resume_path: str,
    reduce_lr: bool = True,
    lr_factor: float = 0.5,
) -> None:
    """Load checkpoint with integrity validation and optional LR reduction.
    
    Args:
        agent: MAPPO agent instance
        resume_path: Path to checkpoint file
        reduce_lr: Whether to reduce learning rate after loading
        lr_factor: Factor to multiply learning rate by (default: 0.5)
    
    Raises:
        RuntimeError: If checkpoint contains NaN/Inf values
    """
    print(f"\n[INFO] Loading model checkpoint from: {resume_path}")
    
    agent.load(resume_path)
    
    # Reduce learning rate when resuming (for curriculum learning stability)
    if reduce_lr and hasattr(agent, 'learning_rate'):
        original_lr = agent.learning_rate
        agent.learning_rate = original_lr * lr_factor
        print(f"[INFO] Reduced LR for stability: {original_lr:.2e} → {agent.learning_rate:.2e}")
    
    # Verify checkpoint integrity
    print("[INFO] Verifying checkpoint integrity...")
    checkpoint_valid = True
    
    for agent_id, model_dict in agent.models.items():
        for role, model in model_dict.items():
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"❌ ERROR: NaN in {agent_id}.{role}.{name}")
                    checkpoint_valid = False
                if torch.isinf(param).any():
                    print(f"❌ ERROR: Inf in {agent_id}.{role}.{name}")
                    checkpoint_valid = False
    
    if not checkpoint_valid:
        raise RuntimeError(
            "❌ Checkpoint contains NaN/Inf! Cannot resume training.\n"
            "   Use a valid checkpoint from an earlier stage."
        )
    
    print("✅ Checkpoint valid - all weights are clean")
    
    # Reset preprocessors (fresh normalization for new curriculum stage)
    preprocessor_attrs = ['state_preprocessor', 'value_preprocessor', 'shared_state_preprocessor']
    for attr in preprocessor_attrs:
        if hasattr(agent, attr):
            preprocessor = getattr(agent, attr)
            if preprocessor is not None:
                preprocessor.reset()
                print(f"[INFO] Reset {attr}")


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
    #num_agents_correct = env.unwrapped.num_agents
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
    
    print("\n[DEBUG] Environment info after SkrlVecEnvWrapper:")
    print(f"Wrapped env type: {type(env)}")
    print(f"Has num_agents: {hasattr(env, 'num_agents')}")
    if hasattr(env, 'num_agents'):
        print(f"Number of agents: {env.num_agents}")
    if hasattr(env, 'possible_agents'):
        print(f"Possible agents: {env.possible_agents}")
    print()

    # ✅ MAPPO/IPPO with custom shared models
    if algorithm in ["mappo", "ippo"] and isinstance(env.unwrapped, DirectMARLEnv):
        
        # Get environment info
        num_agents = len(env.possible_agents)
        obs_space = env.observation_space(env.possible_agents[0])
        act_space = env.action_space(env.possible_agents[0])
        
        # Get state dimension from environment
        dummy_state = env.unwrapped._get_states()
        state_dim = dummy_state.shape[1]
        
        # Create shared models
        models = create_shared_models(
            obs_space=obs_space,
            act_space=act_space,
            state_dim=state_dim,
            num_agents=num_agents,
            possible_agents=env.possible_agents,
            device=env.device,
        )
        
        # Configure and create MAPPO agent
        agent, mappo_cfg = config_mappo_agent(
            env=env,
            agent_cfg=agent_cfg,
            obs_space=obs_space,
            act_space=act_space,
            state_dim=state_dim,
            num_agents=num_agents,
            models=models,
        )
        
        # Load checkpoint if specified
        if resume_path:
            load_checkpoint_with_validation(
                agent=agent,
                resume_path=resume_path,
                reduce_lr=True,
                lr_factor=0.5,
            )
        
        # Create trainer
        trainer_cfg = agent_cfg.get("trainer", {})
        trainer = SequentialTrainer(
            cfg=trainer_cfg,
            env=env,
            agents=agent
        )
        
        # Run training
        print("\n[INFO] Starting training...")
        print("="*80)
        trainer.train()
        
    else:
        # ✅ For non-MARL or PPO, use default Runner
        runner = Runner(env, agent_cfg)
        
        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)
        
        # run training
        print("\n[INFO] Starting training...")
        print("="*80)
        runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
