# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MAPPO training script using TorchRL for Isaac Lab multi-agent UAV swarm.
Simplified to use the MAPPO class from mappo_torchl.py
"""

import argparse
import sys
import torch
import yaml
import gymnasium as gym
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs import TransformedEnv
from tensordict.nn import TensorDictModule
import torch.nn as nn
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="MAPPO training with TorchRL")
parser.add_argument("--task", type=str, default="Isaac-UAV-Swarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg.yaml", 
                    help="Path to config YAML file")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments")
parser.add_argument("--seed", type=int, default=None, help="Override seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--model_name", type=str, default="mappo_uav_swarm", help="Experiment name")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for potential other parsers
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows"""

from torchrl_wrapper import IsaacLabTorchRLWrapper
from mappo_torchl import MAPPOPolicy, CentralizedCritic, MAPPO
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.io import dump_yaml
from datetime import datetime

import UavSwarm.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_policy_module(
    obs_dim: int,
    action_dim: int,
    config: dict,
    device: torch.device,
) -> TensorDictModule:
    """Create policy module wrapped in TensorDictModule with ProbabilisticActor.
    
    This creates the structure expected by TorchRL's ClipPPOLoss.
    """
    # Create the base policy network
    policy_network = MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=config["models"]["policy"]["hidden_sizes"],
    ).to(device)
    
    # Wrap in TensorDictModule to handle TensorDict I/O
    policy_module = TensorDictModule(
        module=policy_network,
        in_keys=[("agents", "observation")],  # Input key from TensorDict
        out_keys=[("agents", "loc"), ("agents", "scale")],  # Output mean and std
    )
    
    # Wrap in ProbabilisticActor to create distributions and sample actions
    policy_actor = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": -1.0,
            "max": 1.0,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    
    return policy_actor


def create_critic_module(
    state_dim: int,
    config: dict,
    device: torch.device,
) -> TensorDictModule:
    """Create centralized critic module wrapped in TensorDictModule."""
    # Create the base critic network
    critic_network = CentralizedCritic(
        state_dim=state_dim,
        hidden_sizes=config["models"]["critic"]["hidden_sizes"],
    ).to(device)
    
    # Wrap in TensorDictModule
    critic_module = TensorDictModule(
        module=critic_network,
        in_keys=[("agents", "observation")],  # Use centralized observation
        out_keys=["state_value"],  # Output value estimate
    )
    
    return critic_module


# Get the agent config entry point (for Hydra)
agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"  # Use MAPPO config entry point


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """Train with TorchRL MAPPO agent."""
    
    # Load TorchRL configuration
    config = load_config(args_cli.config)
    
    # Override env_cfg with CLI arguments and TorchRL config
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else config["env"]["num_envs"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Override seed
    if args_cli.seed is not None:
        config["seed"] = args_cli.seed
    
    # Set seed
    torch.manual_seed(config["seed"])
    env_cfg.seed = config["seed"]
    
    # Setup device
    device = torch.device(config["env"]["device"])
    
    # Setup logging directories (similar to skrl)
    log_root_path = os.path.join("logs", "torchrl", config["training"]["experiment_directory"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_mappo_torchrl"
    if args_cli.model_name:
        log_dir += f'_{args_cli.model_name}'
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Create log directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    
    # Dump configurations
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "torchrl_config.yaml"), config)
    
    # Set log directory in env config
    env_cfg.log_dir = log_dir
    
    print(f"\n{'='*80}")
    print(f"MAPPO Training with TorchRL")
    print(f"{'='*80}")
    print(f"Task: {args_cli.task}")
    print(f"Config: {args_cli.config}")
    print(f"Model name: {args_cli.model_name}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Num envs: {env_cfg.scene.num_envs}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*80}\n")
    
    # Create Isaac Lab environment (following skrl's pattern)
    print("[INFO] Creating Isaac Lab environment...")
    base_env = gym.make(
        args_cli.task,
        cfg=env_cfg,  # ✅ Pass cfg, not num_envs
        render_mode="rgb_array" if args_cli.video else None
    )
    
    # Verify it's a multi-agent environment
    if not isinstance(base_env.unwrapped, DirectMARLEnv):
        raise TypeError(
            f"Environment must be DirectMARLEnv for MAPPO, got {type(base_env.unwrapped)}"
        )
    
    print(f"[INFO] Environment created: {type(base_env.unwrapped).__name__}")
    print(f"[INFO] Number of agents: {base_env.unwrapped.num_agents}")
    print(f"[INFO] Agent IDs: {base_env.unwrapped.possible_agents}\n")
    
    # Wrap for video recording (if enabled)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        base_env = gym.wrappers.RecordVideo(base_env, **video_kwargs)
    
    # Wrap for TorchRL
    print("[INFO] Wrapping environment for TorchRL...")
    env = IsaacLabTorchRLWrapper(
        base_env,
        device=str(device),
        centralized_critic=config["env"]["centralized_critic"]
    )
    
    # Get dimensions
    obs_dim = env.obs_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].shape[0]
    state_dim = env.state_spec["state"].shape[-1]
    num_agents = env.num_agents
    
    print(f"\n[INFO] Environment Dimensions:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Centralized state dim: {state_dim}")
    print(f"  Number of agents: {num_agents}\n")
    
    # Create policy module
    print("[INFO] Creating policy module...")
    policy = create_policy_module(obs_dim, action_dim, config, device)
    
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {param_count:,}")
    
    # Create critic module
    print("[INFO] Creating critic module...")
    critic = create_critic_module(state_dim, config, device)
    
    critic_param_count = sum(p.numel() for p in critic.parameters())
    print(f"  Critic parameters: {critic_param_count:,}\n")
    
    # Create MAPPO trainer
    print("[INFO] Initializing MAPPO trainer...")
    trainer = MAPPO(
        env=env,
        policy=policy,
        critic=critic,
        device=device,
        lr=config["algorithm"]["learning_rate"],
        gamma=config["algorithm"]["gamma"],
        gae_lambda=config["algorithm"]["gae_lambda"],
        clip_epsilon=config["algorithm"]["clip_epsilon"],
        c1=config["algorithm"]["value_loss_coef"],
        c2=config["algorithm"]["entropy_coef"],
        n_epochs=config["algorithm"]["n_epochs"],
        batch_size=config["algorithm"]["batch_size"],
        n_agents=num_agents,
        frames_per_batch=config["algorithm"]["frames_per_batch"],
        model_name=args_cli.model_name,
        checkpoint_interval=config["training"]["checkpoint_interval"],
    )
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"\n[INFO] Loading checkpoint: {args_cli.checkpoint}")
        try:
            checkpoint = torch.load(args_cli.checkpoint, map_location=device)
            
            # Load policy
            if "policy" in checkpoint:
                policy.load_state_dict(checkpoint["policy"])
                print("  ✅ Policy loaded successfully")
            
            # Load critic
            if "critic" in checkpoint:
                critic.load_state_dict(checkpoint["critic"])
                print("  ✅ Critic loaded successfully")
                
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
            print("  Continuing with randomly initialized weights...")
    
    # Start training
    print(f"\n{'='*80}")
    print(f"[INFO] Starting Training")
    print(f"{'='*80}")
    print(f"Total frames: {config['training']['total_frames']:,}")
    print(f"Frames per batch: {config['algorithm']['frames_per_batch']:,}")
    print(f"Checkpoint interval: {config['training']['checkpoint_interval']}")
    print(f"{'='*80}\n")
    
    trainer.train(total_frames=config["training"]["total_frames"])
    
    print(f"\n{'='*80}")
    print(f"[INFO] Training Complete!")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()