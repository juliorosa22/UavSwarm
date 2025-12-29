"""
Script to evaluate individual agent policies in a MAPPO checkpoint and clone the best one.

This script:
1. Loads a MAPPO checkpoint using Runner (same as play.py)
2. Dynamically inspects and extracts individual agent policies
3. Evaluates each agent's policy independently
4. Identifies the best performing policy
5. Creates a new checkpoint with the best policy cloned to all agents
"""

import argparse
import os
import sys
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate and clone best policy from MAPPO checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint to evaluate")
parser.add_argument("--task", type=str, default="Isaac-UAV-Swarm-Direct-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=64, help="Number of evaluation environments")
parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes per agent to evaluate")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for cloned checkpoint")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Name of the RL agent configuration entry point.",
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
    default="MAPPO",
    choices=["MAPPO", "IPPO"],
    help="The RL algorithm used for training.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from pathlib import Path

import skrl
from packaging import version

# Check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import UavSwarm.tasks  # noqa: F401

# Config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent


def inspect_agent_structure(agent):
    """Dynamically inspect agent structure to find policies."""
    print(f"\n{'='*80}")
    print(f"[INFO] Agent Structure Inspection")
    print(f"{'='*80}")
    print(f"Agent type: {type(agent).__name__}")
    print(f"Agent class: {agent.__class__.__module__}.{agent.__class__.__name__}")
    
    # Find all policy-related attributes
    policy_attrs = []
    for attr in dir(agent):
        if not attr.startswith('_') and 'polic' in attr.lower():
            policy_attrs.append(attr)
    
    print(f"\nPolicy-related attributes:")
    for attr in policy_attrs:
        value = getattr(agent, attr)
        print(f"  - {attr}: {type(value)}")
    
    # Check for models attribute (MAPPO stores models here)
    if hasattr(agent, 'models'):
        print(f"\nModels structure:")
        models = agent.models
        print(f"  Type: {type(models)}")
        if isinstance(models, dict):
            print(f"  Keys: {list(models.keys())}")
            for agent_id, agent_models in models.items():
                print(f"    {agent_id}:")
                if isinstance(agent_models, dict):
                    for role, model in agent_models.items():
                        print(f"      {role}: {type(model).__name__}")
                        if hasattr(model, 'net'):
                            print(f"        - has 'net' attribute")
                        if hasattr(model, 'parameters'):
                            param_count = sum(p.numel() for p in model.parameters())
                            print(f"        - parameters: {param_count:,}")
    
    # Check for possible_agents
    if hasattr(agent, 'possible_agents'):
        print(f"\nPossible agents: {agent.possible_agents}")
    
    print(f"{'='*80}\n")
    
    return policy_attrs


def load_checkpoint_with_runner(checkpoint_path: str, env, experiment_cfg: dict):
    """Load checkpoint using Runner (same as play.py)."""
    print(f"\n{'='*80}")
    print(f"[INFO] Loading MAPPO Checkpoint via Runner")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Configure runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    
    # Create runner
    runner = Runner(env, experiment_cfg)
    
    # Load checkpoint
    print(f"[INFO] Loading weights from checkpoint...")
    runner.agent.load(checkpoint_path)
    
    # Set to eval mode
    runner.agent.set_running_mode("eval")
    
    print("✅ Checkpoint loaded successfully via Runner")
    
    # Inspect agent structure
    inspect_agent_structure(runner.agent)
    
    # Validate checkpoint
    print(f"[INFO] Validating checkpoint integrity...")
    checkpoint_valid = validate_checkpoint_integrity(runner.agent)
    
    if not checkpoint_valid:
        raise RuntimeError("Checkpoint contains invalid weights (NaN/Inf)")
    
    print("✅ Checkpoint validated")
    print(f"{'='*80}\n")
    
    return runner


def validate_checkpoint_integrity(agent) -> bool:
    """Validate that checkpoint doesn't contain NaN/Inf."""
    nan_count = 0
    inf_count = 0
    
    # Check all models
    if hasattr(agent, 'models') and isinstance(agent.models, dict):
        for agent_id, agent_models in agent.models.items():
            if isinstance(agent_models, dict):
                for role, model in agent_models.items():
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            nan_count += torch.isnan(param).sum().item()
                            print(f"  ❌ NaN in {agent_id}.{role}.{name}")
                        if torch.isinf(param).any():
                            inf_count += torch.isinf(param).sum().item()
                            print(f"  ❌ Inf in {agent_id}.{role}.{name}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"  ❌ Found {nan_count} NaN and {inf_count} Inf values")
        return False
    
    print(f"  ✅ All weights are valid")
    return True


def extract_policy_for_agent(agent, agent_id: str):
    """Extract policy network for a specific agent."""
    if hasattr(agent, 'models') and isinstance(agent.models, dict):
        if agent_id in agent.models:
            agent_models = agent.models[agent_id]
            if isinstance(agent_models, dict) and 'policy' in agent_models:
                return agent_models['policy']
    
    raise ValueError(f"Could not find policy for agent '{agent_id}'")


def evaluate_single_policy(
    env,
    runner,
    eval_agent_id: str,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate a single agent's policy by using it for ALL agents in the environment."""
    print(f"\n[INFO] Evaluating policy of {eval_agent_id}...")
    
    # Extract the policy to evaluate
    eval_policy = extract_policy_for_agent(runner.agent, eval_agent_id)
    
    print(f"  Policy type: {type(eval_policy).__name__}")
    print(f"  Policy parameters: {sum(p.numel() for p in eval_policy.parameters()):,}")
    
    # ✅ Test the policy output format first
    print(f"  [DEBUG] Testing policy output format...")
    with torch.inference_mode():
        dummy_obs = torch.zeros(1, env.observation_space(env.possible_agents[0]).shape[0], device=env.device)
        dummy_input = {"states": dummy_obs}
        test_output = eval_policy.act(dummy_input, role="policy")
        
        print(f"  [DEBUG] act() returns: {type(test_output)}")
        if isinstance(test_output, tuple):
            print(f"  [DEBUG] Tuple length: {len(test_output)}")
            print(f"  [DEBUG] Element types: {[type(x) for x in test_output]}")
            print(f"  [DEBUG] Element 0 shape: {test_output[0].shape if hasattr(test_output[0], 'shape') else 'N/A'}")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        timestep = 0
        episode_reward = 0.0
        
        while not done and timestep < 1000:
            with torch.inference_mode():
                actions = {}
                
                for agent_id in env.possible_agents:
                    policy_input = {"states": obs[agent_id]}
                    
                    # ✅ ROBUST: Handle any return format
                    try:
                        result = eval_policy.act(policy_input, role="policy")
                        
                        # Extract action regardless of return format
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
                        
                        # Optional: Try to get mean for deterministic actions
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
                        print(f"  ❌ Fatal error getting action for {agent_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Emergency fallback
                        action_dim = env.action_space(agent_id).shape[0]
                        actions[agent_id] = torch.zeros(
                            obs[agent_id].shape[0], 
                            action_dim, 
                            device=env.device
                        )
                
                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Accumulate rewards
                episode_reward += sum(r.mean().item() for r in rewards.values()) / len(rewards)
                
                # Check if done
                done = all(t.all().item() for t in terminated.values()) or \
                       all(t.all().item() for t in truncated.values())
                
                timestep += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(timestep)
        
        print(f"  Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, Length={timestep}")
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }
    
    return metrics


def compare_all_policies(
    env,
    runner,
    num_episodes: int = 10,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Evaluate all agent policies and compare them.
    
    Returns:
        best_agent_id: ID of the best performing agent
        all_metrics: Dictionary mapping agent_id to metrics
    """
    print(f"\n{'='*80}")
    print(f"[INFO] Comparing All Agent Policies")
    print(f"{'='*80}")
    
    # Get list of agents
    if hasattr(runner.agent, 'possible_agents'):
        agent_ids = runner.agent.possible_agents
    elif hasattr(runner.agent, 'models'):
        agent_ids = list(runner.agent.models.keys())
    else:
        raise ValueError("Could not determine agent IDs from runner.agent")
    
    print(f"Agents to evaluate: {len(agent_ids)}")
    print(f"Episodes per agent: {num_episodes}")
    print(f"Evaluation environments: {env.num_envs}")
    print()
    
    all_metrics = {}
    
    for agent_id in agent_ids:
        metrics = evaluate_single_policy(env, runner, agent_id, num_episodes)
        all_metrics[agent_id] = metrics
        
        print(f"\n[RESULTS] {agent_id}:")
        print(f"  Mean Reward:  {metrics['mean_reward']:8.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Min/Max:      {metrics['min_reward']:8.2f} / {metrics['max_reward']:8.2f}")
        print(f"  Mean Length:  {metrics['mean_length']:8.1f} steps")
    
    # Find best agent
    best_agent_id = max(all_metrics.keys(), key=lambda k: all_metrics[k]['mean_reward'])
    
    print(f"\n{'='*80}")
    print(f"[WINNER] Best Policy: {best_agent_id}")
    print(f"{'='*80}")
    print(f"Mean Reward: {all_metrics[best_agent_id]['mean_reward']:.2f}")
    print(f"Std Reward:  {all_metrics[best_agent_id]['std_reward']:.2f}")
    print(f"{'='*80}\n")
    
    return best_agent_id, all_metrics


def clone_best_policy(
    runner,
    best_agent_id: str,
    output_path: str,
) -> None:
    """Clone the best policy to all agents and save new checkpoint.
    
    Args:
        runner: Runner instance with loaded agent
        best_agent_id: ID of best performing agent
        output_path: Path to save cloned checkpoint
    """
    print(f"\n{'='*80}")
    print(f"[INFO] Cloning Best Policy to All Agents")
    print(f"{'='*80}")
    print(f"Best policy: {best_agent_id}")
    print(f"Output: {output_path}")
    
    agent = runner.agent
    
    # Get best policy
    best_policy = extract_policy_for_agent(agent, best_agent_id)
    
    # Get all agent IDs
    if hasattr(agent, 'possible_agents'):
        agent_ids = agent.possible_agents
    else:
        agent_ids = list(agent.models.keys())
    
    # Clone to all agents
    for agent_id in agent_ids:
        if agent_id == best_agent_id:
            continue  # Skip the source agent
        
        # Get current policy
        current_policy = extract_policy_for_agent(agent, agent_id)
        
        # Copy state dict (weights and biases)
        current_policy.load_state_dict(best_policy.state_dict())
        
        print(f"  ✅ Cloned {best_agent_id} → {agent_id}")
    
    # Save new checkpoint
    print(f"\n[INFO] Saving cloned checkpoint...")
    agent.save(output_path)
    print(f"✅ Checkpoint saved to: {output_path}")
    
    # Verify cloning
    print(f"\n[INFO] Verifying weight cloning...")
    all_same = True
    for agent_id in agent_ids:
        if agent_id == best_agent_id:
            continue
        
        policy = extract_policy_for_agent(agent, agent_id)
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            best_policy.named_parameters(),
            policy.named_parameters()
        ):
            if not torch.allclose(param1, param2, atol=1e-6):
                print(f"  ❌ {agent_id}.{name2} differs from {best_agent_id}.{name1}")
                all_same = False
                break
    
    if all_same:
        print(f"✅ All agents now have identical policies (cloned from {best_agent_id})")
    else:
        print(f"⚠️  Warning: Some policies may not have been cloned correctly")
    
    print(f"{'='*80}\n")


def generate_comparison_report(
    all_metrics: Dict[str, Dict[str, float]],
    best_agent_id: str,
    output_dir: str,
) -> None:
    """Generate a detailed comparison report and save to file."""
    report_path = os.path.join(output_dir, "policy_comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MAPPO Policy Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Agent ID':<15} {'Mean Reward':>12} {'Std Reward':>12} "
                f"{'Min Reward':>12} {'Max Reward':>12} {'Avg Length':>12}\n")
        f.write("-"*80 + "\n")
        
        for agent_id in sorted(all_metrics.keys()):
            metrics = all_metrics[agent_id]
            marker = "  ⭐ BEST" if agent_id == best_agent_id else ""
            f.write(f"{agent_id:<15} "
                   f"{metrics['mean_reward']:>12.2f} "
                   f"{metrics['std_reward']:>12.2f} "
                   f"{metrics['min_reward']:>12.2f} "
                   f"{metrics['max_reward']:>12.2f} "
                   f"{metrics['mean_length']:>12.1f}"
                   f"{marker}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Performance differences
        f.write("Performance Differences from Best:\n")
        f.write("-"*80 + "\n")
        best_reward = all_metrics[best_agent_id]['mean_reward']
        
        for agent_id in sorted(all_metrics.keys()):
            if agent_id == best_agent_id:
                continue
            
            diff = all_metrics[agent_id]['mean_reward'] - best_reward
            pct_diff = (diff / abs(best_reward)) * 100 if best_reward != 0 else 0
            
            f.write(f"{agent_id}: {diff:+.2f} ({pct_diff:+.1f}%)\n")
        
        f.write("-"*80 + "\n\n")
        
        # Recommendation
        f.write("Recommendation:\n")
        f.write("-"*80 + "\n")
        f.write(f"Clone policy from {best_agent_id} to all agents for improved performance.\n")
        f.write(f"Expected improvement: {best_reward:.2f} avg reward\n")
        f.write("-"*80 + "\n")
    
    print(f"[INFO] Detailed report saved to: {report_path}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, experiment_cfg: dict):
    """Main evaluation and cloning pipeline."""
    
    # Setup
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed
    
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine output directory
    if args_cli.output_dir:
        output_dir = os.path.abspath(args_cli.output_dir)
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_dir = os.path.join(checkpoint_dir, "cloned_policies")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MAPPO Policy Evaluation & Cloning Pipeline (Runner-based)")
    print(f"{'='*80}")
    print(f"Input checkpoint:  {checkpoint_path}")
    print(f"Output directory:  {output_dir}")
    print(f"Evaluation episodes: {args_cli.eval_episodes} per agent")
    print(f"{'='*80}\n")
    
    # Create environment
    print("[INFO] Creating evaluation environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise TypeError("This script requires a DirectMARLEnv (multi-agent environment)")
    
    print(f"✅ Environment created: {len(env.unwrapped.possible_agents)} agents")
    
    # Wrap environment
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    
    # Load checkpoint using Runner
    runner = load_checkpoint_with_runner(checkpoint_path, env, experiment_cfg)
    
    # Evaluate all policies
    best_agent_id, all_metrics = compare_all_policies(
        env,
        runner,
        num_episodes=args_cli.eval_episodes,
    )
    
    # Generate report
    generate_comparison_report(all_metrics, best_agent_id, output_dir)
    
    # Clone best policy
    cloned_checkpoint_path = os.path.join(
        output_dir,
        f"agent_cloned_from_{best_agent_id}.pt"
    )
    clone_best_policy(runner, best_agent_id, cloned_checkpoint_path)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Pipeline Complete!")
    print(f"{'='*80}")
    print(f"Best policy: {best_agent_id}")
    print(f"Mean reward: {all_metrics[best_agent_id]['mean_reward']:.2f}")
    print(f"Cloned checkpoint: {cloned_checkpoint_path}")
    print(f"Comparison report: {os.path.join(output_dir, 'policy_comparison_report.txt')}")
    print(f"{'='*80}\n")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()