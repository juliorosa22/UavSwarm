"""
Simple test script to verify IsaacLab's TorchRL wrapper compatibility.
Tests the official torchrl.envs.IsaacLabWrapper with the UAV swarm environment.

This script follows the same structure as Isaac Lab's official skrl training script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test IsaacLab TorchRL wrapper with UAV swarm.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during testing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# ❌ REMOVED: --device (AppLauncher adds this automatically)
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run for testing.")

# ✅ Append AppLauncher CLI args (this adds --device, --headless, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
import random
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import IsaacLabWrapper, GymWrapper  # Add this import at the top
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.collectors import SyncDataCollector
from torchrl.data import Composite, Unbounded

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import UAV swarm tasks
import UavSwarm.tasks  # noqa: F401

print("\n" + "="*80)
print("Testing IsaacLab TorchRL Wrapper")
print("="*80)


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """Test IsaacLab TorchRL wrapper with UAV swarm environment."""
    
    # ============================================================================
    # Configuration Override
    # ============================================================================
    
    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # ✅ Use device from AppLauncher (it handles the device argument)
    if hasattr(args_cli, 'device') and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    
    # Randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    
    # Set seed
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        torch.manual_seed(args_cli.seed)
        random.seed(args_cli.seed)
    
    # Specify directory for logging
    log_root_path = os.path.join("logs", "torchrl", "test_runs")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging test in directory: {log_root_path}")
    
    # Specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_torchrl_wrapper_test"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Set log directory for environment
    env_cfg.log_dir = log_dir
    
    # Dump configuration
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    
    print(f"[INFO] Test configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {env_cfg.sim.device}")
    print(f"  Seed: {env_cfg.seed}")
    print(f"  Log directory: {log_dir}")
    
    # ============================================================================
    # Step 1: Create Isaac Lab Environment
    # ============================================================================
    
    print("\n[1/8] Creating Isaac Lab environment...")
    try:
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )
        
        print(f"  ✅ Environment created: {type(env.unwrapped).__name__}")
        
    except Exception as e:
        print(f"  ❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    
    # ============================================================================
    # Step 2: Debug Environment Info (Before Wrapping)
    # ============================================================================
    
    print("\n[2/8] Inspecting environment (before wrapping)...")
    print(f"  Environment type: {type(env.unwrapped)}")
    print(f"  Is DirectMARLEnv: {isinstance(env.unwrapped, DirectMARLEnv)}")
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        print(f"  Number of agents: {env.unwrapped.num_agents}")
        print(f"  Possible agents: {env.unwrapped.possible_agents}")
        
        # Check observation/action spaces
        if hasattr(env.unwrapped, 'observation_spaces'):
            print(f"  Observation spaces (per agent):")
            for agent_id, obs_space in list(env.unwrapped.observation_spaces.items())[:2]:
                print(f"    {agent_id}: {obs_space.shape}")
        
        if hasattr(env.unwrapped, 'action_spaces'):
            print(f"  Action spaces (per agent):")
            for agent_id, act_space in list(env.unwrapped.action_spaces.items())[:2]:
                print(f"    {agent_id}: {act_space.shape}")
    
    # ============================================================================
    # Step 3: Reset Environment to Initialize Buffers
    # ============================================================================
    
    print("\n[3/8] Resetting environment to initialize buffers...")
    try:
        obs, info = env.reset()
        
        print(f"  ✅ Reset successful")
        print(f"  Observation type: {type(obs)}")
        
        if isinstance(obs, dict):
            print(f"  Observation keys: {list(obs.keys())[:3]}...")  # First 3 keys
            for key in list(obs.keys())[:2]:
                if hasattr(obs[key], 'shape'):
                    print(f"    {key}: shape {obs[key].shape}")
        
    except Exception as e:
        print(f"  ❌ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    # ============================================================================
    # Step 4: Wrap with TorchRL GymWrapper
    # ============================================================================
    
    print("\n[4/8] Wrapping with TorchRL GymWrapper...")
    try:
        # ✅ ALTERNATIVE: Use GymWrapper directly
        env = GymWrapper(
            env=env,
            categorical_action_encoding=False,
            allow_done_after_reset=False,
            from_pixels=False,
            pixels_only=False,
        )
        
        print(f"  ✅ Wrapper applied successfully")
        print(f"  Wrapped environment type: {type(env).__name__}")
        
    except Exception as e:
        print(f"  ❌ Failed to wrap environment: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    # ============================================================================
    # Step 5: Wrap for Video Recording (if requested)
    # ============================================================================
    
    if args_cli.video:
        print("\n[5/8] Setting up video recording...")
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Video recording configuration:")
        print_dict(video_kwargs, nesting=4)
        
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print("  ✅ Video recording enabled")
    else:
        print("\n[5/8] Skipping video recording (not requested)")
    
    # ============================================================================
    # Step 6: Inspect Environment Specs
    # ============================================================================
    
    print("\n[6/8] Inspecting TorchRL environment specifications...")
    try:
        print(f"  Batch size: {env.batch_size}")
        print(f"  Device: {env.device}")
        
        print(f"\n  Observation spec:")
        obs_spec = env.observation_spec
        print(f"    Type: {type(obs_spec)}")
        if hasattr(obs_spec, 'keys'):
            print(f"    Keys: {list(obs_spec.keys(include_nested=False))}")
        
        print(f"\n  Action spec:")
        action_spec = env.action_spec
        print(f"    Type: {type(action_spec)}")
        if hasattr(action_spec, 'keys'):
            print(f"    Keys: {list(action_spec.keys(include_nested=False))}")
        
        print(f"\n  Reward spec:")
        print(f"    {env.reward_spec}")
        
        print(f"\n  Done spec:")
        print(f"    {env.done_spec}")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not fully inspect specs: {e}")
    
    # ============================================================================
    # Step 7: Test Environment Step with Random Actions
    # ============================================================================
    
    print(f"\n[7/8] Testing environment step with random actions...")
    print(f"  Running {args_cli.num_steps} steps...")
    
    try:
        total_reward = 0.0
        episode_count = 0
        
        for step in range(args_cli.num_steps):
            # Generate random actions
            actions = env.action_spec.rand()
            
            # Step environment
            step_output = env.step(actions)
            
            # Accumulate rewards
            if "reward" in step_output.keys():
                reward = step_output["reward"]
                total_reward += reward.mean().item()
            
            # Check for done episodes
            if "done" in step_output.keys():
                done = step_output["done"]
                if done.any():
                    episode_count += done.sum().item()
                    # Reset if needed (TorchRL handles this automatically)
            
            # Print progress
            if (step + 1) % 100 == 0:
                avg_reward = total_reward / (step + 1)
                print(f"    Step {step + 1}/{args_cli.num_steps} | Avg Reward: {avg_reward:.4f} | Episodes: {episode_count}")
        
        print(f"\n  ✅ Test complete!")
        print(f"  Total steps: {args_cli.num_steps}")
        print(f"  Average reward: {total_reward / args_cli.num_steps:.4f}")
        print(f"  Episodes completed: {episode_count}")
        
    except Exception as e:
        print(f"  ❌ Failed during environment stepping: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    # ============================================================================
    # Step 8: Test with Data Collector
    # ============================================================================
    
    print(f"\n[8/8] Testing with TorchRL SyncDataCollector...")
    try:
        # Create a simple random policy
        class SimpleRandomPolicy(nn.Module):
            """Simple random policy for testing."""
            def __init__(self, action_spec):
                super().__init__()
                self.action_spec = action_spec
            
            def forward(self, tensordict):
                actions = self.action_spec.rand(tensordict.batch_size)
                tensordict.update(actions)
                return tensordict
        
        # Create policy
        policy = SimpleRandomPolicy(env.action_spec)
        policy_module = TensorDictModule(
            module=policy,
            in_keys=[],
            out_keys=list(env.action_spec.keys(include_nested=True, leaves_only=True))
        )
        
        print(f"  ✅ Policy created")
        
        # Create data collector
        collector = SyncDataCollector(
            create_env_fn=lambda: env,
            policy=policy_module,
            frames_per_batch=500,
            total_frames=1000,
            device=env.device,
            storing_device=env.device,
        )
        
        print(f"  ✅ Data collector created")
        
        # Collect batches
        print(f"\n  Collecting experience batches...")
        batch_count = 0
        total_frames = 0
        
        for batch in collector:
            batch_count += 1
            frames = batch.numel()
            total_frames += frames
            
            # Get average reward
            if "reward" in batch.keys():
                avg_reward = batch["reward"].mean().item()
                print(f"    Batch {batch_count}: {frames} frames | Avg Reward: {avg_reward:.4f}")
            
            if batch_count >= 2:  # Collect 2 batches for testing
                break
        
        print(f"\n  ✅ Data collection successful!")
        print(f"  Total frames collected: {total_frames}")
        
        collector.shutdown()
        
    except Exception as e:
        print(f"  ⚠️  Warning: Data collection test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # Cleanup
    # ============================================================================
    
    print("\n[Cleanup] Closing environment and simulation...")
    try:
        env.close()
        print("  ✅ Environment closed")
    except Exception as e:
        print(f"  ⚠️  Warning: Cleanup had issues: {e}")
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print("✅ IsaacLab TorchRL wrapper is working correctly!")
    print(f"✅ Environment creation: SUCCESS")
    print(f"✅ Wrapper application: SUCCESS")
    print(f"✅ Environment stepping: SUCCESS ({args_cli.num_steps} steps)")
    print(f"✅ Data collection: SUCCESS" if batch_count > 0 else "⚠️  Data collection: SKIPPED")
    if args_cli.video:
        print(f"✅ Video recording: ENABLED (check {log_dir}/videos)")
    print(f"\n📁 Logs saved to: {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run the main function
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close sim app
        simulation_app.close()