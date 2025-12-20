"""
Script to inspect skrl model checkpoints for NaN/Inf values, weight statistics,
and verify if policy weights are shared across agents.

Usage:
    python scripts/check_trained_model.py --checkpoint logs/skrl/run_name/checkpoints/agent.pt
    python scripts/check_trained_model.py --log_dir logs/skrl/run_name  # Auto-find checkpoints
    python scripts/check_trained_model.py --checkpoint agent.pt --check_sharing  # Verify weight sharing
"""

import argparse
import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def find_checkpoints(log_dir: str) -> List[str]:
    """Find all checkpoint files in a log directory."""
    checkpoint_paths = []
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"[ERROR] Directory not found: {log_dir}")
        return []
    
    # Search for checkpoint files
    for checkpoint_file in log_path.rglob("*.pt"):
        if "checkpoint" in checkpoint_file.parent.name.lower() or "agent" in checkpoint_file.name:
            checkpoint_paths.append(str(checkpoint_file))
    
    return sorted(checkpoint_paths)


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint file safely."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None


def inspect_tensor_stats(name: str, tensor: torch.Tensor) -> Dict:
    """Compute statistics for a tensor."""
    stats = {
        "name": name,
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype),
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "num_nan": torch.isnan(tensor).sum().item() if torch.isnan(tensor).any() else 0,
        "num_inf": torch.isinf(tensor).sum().item() if torch.isinf(tensor).any() else 0,
    }
    
    # Only compute stats if no NaN/Inf
    if not stats["has_nan"] and not stats["has_inf"]:
        stats.update({
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "abs_mean": tensor.abs().mean().item(),
            "abs_max": tensor.abs().max().item(),
        })
    else:
        stats.update({
            "min": float('nan'),
            "max": float('nan'),
            "mean": float('nan'),
            "std": float('nan'),
            "abs_mean": float('nan'),
            "abs_max": float('nan'),
        })
    
    return stats


def extract_agent_weights(agent_data: Dict, agent_name: str) -> Dict[str, torch.Tensor]:
    """Extract weights from agent data, searching common locations."""
    weights = {}
    
    # Common locations for weights in skrl checkpoints
    weight_locations = [
        ("direct", agent_data),
        ("policy", agent_data.get('policy', {})),
        ("policy.state_dict", agent_data.get('policy', {}).get('state_dict', {})),
        ("state_dict", agent_data.get('state_dict', {})),
        ("value", agent_data.get('value', {})),
        ("value.state_dict", agent_data.get('value', {}).get('state_dict', {})),
    ]
    
    for location_name, location in weight_locations:
        if not isinstance(location, dict):
            continue
        
        for param_name, param_tensor in location.items():
            if isinstance(param_tensor, torch.Tensor):
                full_name = f"{location_name}.{param_name}" if location_name != "direct" else param_name
                weights[full_name] = param_tensor
    
    return weights


def check_weight_sharing(checkpoint: Dict) -> Dict:
    """Check if agents share policy weights or have separate policies.
    
    Returns:
        Dict with sharing analysis results
    """
    print(f"\n{'='*80}")
    print(f"WEIGHT SHARING ANALYSIS")
    print(f"{'='*80}")
    
    # Get all agent keys (assume they're named like 'robot_0', 'robot_1', etc.)
    agent_keys = [k for k in checkpoint.keys() if 'robot' in k.lower() or 'agent' in k.lower()]
    
    if len(agent_keys) < 2:
        print(f"[INFO] Only {len(agent_keys)} agent(s) found - cannot check weight sharing")
        return {"num_agents": len(agent_keys), "shared_policy": None}
    
    print(f"[INFO] Found {len(agent_keys)} agents: {agent_keys[:5]}{'...' if len(agent_keys) > 5 else ''}")
    
    # Extract weights for each agent
    agent_weights = {}
    for agent_key in agent_keys:
        weights = extract_agent_weights(checkpoint[agent_key], agent_key)
        if weights:
            agent_weights[agent_key] = weights
    
    if len(agent_weights) < 2:
        print(f"[WARNING] Could not extract weights from enough agents")
        return {"num_agents": len(agent_keys), "shared_policy": None}
    
    # Compare policy weights between agents
    reference_agent = agent_keys[0]
    reference_weights = agent_weights[reference_agent]
    
    print(f"\n[INFO] Using '{reference_agent}' as reference agent")
    print(f"[INFO] Found {len(reference_weights)} parameter tensors")
    
    # Filter for policy-related weights
    policy_weight_names = [
        name for name in reference_weights.keys()
        if any(keyword in name.lower() for keyword in ['policy', 'actor', 'net', 'mean', 'std'])
        and 'value' not in name.lower() and 'critic' not in name.lower()
    ]
    
    if not policy_weight_names:
        print(f"[WARNING] Could not identify policy weights (searched for 'policy', 'actor', 'net')")
        print(f"[INFO] Available weight names: {list(reference_weights.keys())[:10]}")
        # Fallback: use all weights
        policy_weight_names = list(reference_weights.keys())
    
    print(f"[INFO] Identified {len(policy_weight_names)} policy-related parameters:")
    for name in policy_weight_names[:10]:
        print(f"  - {name}")
    if len(policy_weight_names) > 10:
        print(f"  ... and {len(policy_weight_names) - 10} more")
    
    # Compare each policy weight across agents
    print(f"\n{'─'*80}")
    print(f"Comparing policy weights across agents")
    print(f"{'─'*80}")
    
    shared_weights = {}
    separate_weights = {}
    
    for weight_name in policy_weight_names:
        reference_tensor = reference_weights[weight_name]
        
        # Compare with all other agents
        all_identical = True
        max_diff = 0.0
        
        for other_agent in agent_keys[1:]:
            if other_agent not in agent_weights:
                continue
            
            other_weights = agent_weights[other_agent]
            
            if weight_name not in other_weights:
                print(f"  ⚠️  '{weight_name}' not found in {other_agent}")
                all_identical = False
                break
            
            other_tensor = other_weights[weight_name]
            
            # Check if shapes match
            if reference_tensor.shape != other_tensor.shape:
                print(f"  ⚠️  Shape mismatch for '{weight_name}': {reference_tensor.shape} vs {other_tensor.shape}")
                all_identical = False
                break
            
            # Compute difference
            diff = torch.abs(reference_tensor - other_tensor).max().item()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-6:  # Threshold for considering weights different
                all_identical = False
        
        # Categorize
        if all_identical:
            shared_weights[weight_name] = {
                "shape": reference_tensor.shape,
                "max_diff": max_diff,
                "mean": reference_tensor.mean().item(),
            }
        else:
            separate_weights[weight_name] = {
                "shape": reference_tensor.shape,
                "max_diff": max_diff,
            }
    
    # Print results
    print(f"\n{'─'*80}")
    print(f"RESULTS")
    print(f"{'─'*80}")
    
    if shared_weights:
        print(f"\n✅ SHARED POLICY WEIGHTS (identical across all agents): {len(shared_weights)}")
        for name, info in list(shared_weights.items())[:5]:
            print(f"  - {name}: shape={info['shape']}, max_diff={info['max_diff']:.2e}")
        if len(shared_weights) > 5:
            print(f"  ... and {len(shared_weights) - 5} more")
    
    if separate_weights:
        print(f"\n❌ SEPARATE POLICY WEIGHTS (different per agent): {len(separate_weights)}")
        for name, info in list(separate_weights.items())[:5]:
            print(f"  - {name}: shape={info['shape']}, max_diff={info['max_diff']:.2e}")
        if len(separate_weights) > 5:
            print(f"  ... and {len(separate_weights) - 5} more")
    
    # Final verdict
    print(f"\n{'='*80}")
    total_policy_weights = len(shared_weights) + len(separate_weights)
    shared_percentage = 100 * len(shared_weights) / total_policy_weights if total_policy_weights > 0 else 0
    
    if shared_percentage > 95:
        print(f"✅ VERDICT: SHARED POLICY")
        print(f"   All agents use the SAME policy network weights ({shared_percentage:.1f}% identical)")
        print(f"   This is correct for homogeneous multi-agent learning!")
        shared_policy = True
    elif shared_percentage < 5:
        print(f"❌ VERDICT: SEPARATE POLICIES")
        print(f"   Each agent has INDEPENDENT policy weights ({100-shared_percentage:.1f}% different)")
        print(f"   This means agents are NOT sharing weights (might be intentional or a bug)")
        shared_policy = False
    else:
        print(f"⚠️  VERDICT: PARTIALLY SHARED")
        print(f"   {shared_percentage:.1f}% of weights are shared, {100-shared_percentage:.1f}% are separate")
        print(f"   This is unusual - check your model configuration!")
        shared_policy = None
    
    print(f"{'='*80}")
    
    return {
        "num_agents": len(agent_keys),
        "shared_policy": shared_policy,
        "num_shared_weights": len(shared_weights),
        "num_separate_weights": len(separate_weights),
        "shared_percentage": shared_percentage,
    }


def inspect_checkpoint(checkpoint_path: str, verbose: bool = True, check_sharing: bool = False) -> Dict:
    """Inspect a single checkpoint file."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Check file exists and size
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] File not found: {checkpoint_path}")
        return None
    
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 0.1:
        print(f"[WARNING] File seems too small ({file_size_mb:.2f} MB) - might be corrupted!")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        return None
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    # ✅ Deep inspection of nested structure
    print(f"\n{'─'*80}")
    print(f"Checkpoint Structure Deep Dive")
    print(f"{'─'*80}")
    
    for key in checkpoint.keys():
        value = checkpoint[key]
        print(f"\n'{key}': type={type(value)}")
        
        if isinstance(value, dict):
            print(f"  Dict with keys: {list(value.keys())}")
            
            # Check for common skrl checkpoint structures
            for sub_key in ['policy', 'value', 'optimizer', 'scheduler', 'state_dict']:
                if sub_key in value:
                    sub_value = value[sub_key]
                    print(f"    '{sub_key}': type={type(sub_value)}")
                    
                    if isinstance(sub_value, dict):
                        print(f"      Sub-keys: {list(sub_value.keys())[:10]}")  # First 10 keys
                        
                        # Count tensors
                        tensor_count = sum(1 for v in sub_value.values() if isinstance(v, torch.Tensor))
                        print(f"      Tensors found: {tensor_count}")
                    elif isinstance(sub_value, torch.Tensor):
                        print(f"      Tensor shape: {sub_value.shape}")
        
        elif isinstance(value, torch.Tensor):
            print(f"  Tensor shape: {value.shape}")
        
        elif isinstance(value, (int, float, str)):
            print(f"  Value: {value}")
    
    # ✅ NEW: Check weight sharing if requested
    if check_sharing:
        sharing_results = check_weight_sharing(checkpoint)
    else:
        sharing_results = None
    
    # Results storage
    results = {
        "path": checkpoint_path,
        "file_size_mb": file_size_mb,
        "agents": {},
        "has_any_nan": False,
        "has_any_inf": False,
        "sharing_analysis": sharing_results,
    }
    
    # Inspect each agent's weights
    for agent_key in checkpoint.keys():
        if not isinstance(checkpoint[agent_key], dict):
            continue
        
        print(f"\n{'─'*80}")
        print(f"Agent: {agent_key}")
        print(f"{'─'*80}")
        
        agent_stats = {
            "total_params": 0,
            "total_nan": 0,
            "total_inf": 0,
            "layers": [],
        }
        
        # Extract and inspect weights
        weights = extract_agent_weights(checkpoint[agent_key], agent_key)
        
        weights_found = len(weights) > 0
        
        for param_name, param_tensor in weights.items():
            stats = inspect_tensor_stats(param_name, param_tensor)
            agent_stats["layers"].append(stats)
            agent_stats["total_params"] += stats["numel"]
            agent_stats["total_nan"] += stats["num_nan"]
            agent_stats["total_inf"] += stats["num_inf"]
            
            # Print layer info
            if verbose or stats["has_nan"] or stats["has_inf"]:
                status = "✅" if not (stats["has_nan"] or stats["has_inf"]) else "❌"
                print(f"\n  {status} {param_name}")
                print(f"    Shape: {stats['shape']}, Params: {stats['numel']:,}")
                
                if stats["has_nan"]:
                    print(f"    ❌ NaN values: {stats['num_nan']:,} ({100*stats['num_nan']/stats['numel']:.2f}%)")
                
                if stats["has_inf"]:
                    print(f"    ❌ Inf values: {stats['num_inf']:,} ({100*stats['num_inf']/stats['numel']:.2f}%)")
                
                if not stats["has_nan"] and not stats["has_inf"]:
                    print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                    print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                    print(f"    Abs Mean: {stats['abs_mean']:.6f}, Abs Max: {stats['abs_max']:.6f}")
                    
                    # Flag suspicious values
                    if stats['abs_max'] > 1000:
                        print(f"    ⚠️  Large weights detected (max abs: {stats['abs_max']:.2f})")
                    if stats['abs_mean'] < 1e-6:
                        print(f"    ⚠️  Very small weights (mean abs: {stats['abs_mean']:.2e})")
        
        if not weights_found:
            print(f"\n  ⚠️  WARNING: No tensor parameters found for agent '{agent_key}'!")
            print(f"  Available keys: {list(checkpoint[agent_key].keys())}")
        
        # Summary for this agent
        print(f"\n  Summary:")
        print(f"    Total parameters: {agent_stats['total_params']:,}")
        print(f"    Total NaN: {agent_stats['total_nan']:,}")
        print(f"    Total Inf: {agent_stats['total_inf']:,}")
        
        if not weights_found:
            print(f"    ❌ NO WEIGHTS FOUND - CHECKPOINT IS EMPTY OR WRONG FORMAT!")
        elif agent_stats['total_nan'] > 0 or agent_stats['total_inf'] > 0:
            print(f"    ⚠️  MODEL IS CORRUPTED!")
            results["has_any_nan"] = results["has_any_nan"] or (agent_stats['total_nan'] > 0)
            results["has_any_inf"] = results["has_any_inf"] or (agent_stats['total_inf'] > 0)
        else:
            print(f"    ✅ Model weights are valid!")
        
        results["agents"][agent_key] = agent_stats
    
    return results


def compare_checkpoints(checkpoint_paths: List[str]) -> None:
    """Compare multiple checkpoints to see when corruption started."""
    print(f"\n{'='*80}")
    print(f"Comparing {len(checkpoint_paths)} checkpoints")
    print(f"{'='*80}")
    
    results = []
    for checkpoint_path in checkpoint_paths:
        result = inspect_checkpoint(checkpoint_path, verbose=False, check_sharing=False)
        if result:
            results.append(result)
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"Checkpoint Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<50} {'Size (MB)':<12} {'NaN':<8} {'Inf':<8} {'Status'}")
    print(f"{'─'*50} {'─'*12} {'─'*8} {'─'*8} {'─'*10}")
    
    for result in results:
        checkpoint_name = Path(result["path"]).name
        has_nan = "❌" if result["has_any_nan"] else "✅"
        has_inf = "❌" if result["has_any_inf"] else "✅"
        status = "CORRUPTED" if (result["has_any_nan"] or result["has_any_inf"]) else "Valid"
        
        print(f"{checkpoint_name:<50} {result['file_size_mb']:<12.2f} {has_nan:<8} {has_inf:<8} {status}")
    
    # Find first corrupted checkpoint
    first_corrupted_idx = None
    for i, result in enumerate(results):
        if result["has_any_nan"] or result["has_any_inf"]:
            first_corrupted_idx = i
            break
    
    if first_corrupted_idx is not None:
        print(f"\n⚠️  First corrupted checkpoint: {Path(results[first_corrupted_idx]['path']).name}")
        if first_corrupted_idx > 0:
            print(f"✅ Last valid checkpoint: {Path(results[first_corrupted_idx - 1]['path']).name}")
            print(f"\n💡 Recommendation: Use {Path(results[first_corrupted_idx - 1]['path']).name} for inference")
    else:
        print(f"\n✅ All checkpoints are valid!")


def main():
    parser = argparse.ArgumentParser(description="Inspect skrl model checkpoints for corruption and weight sharing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file (.pt)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Path to log directory (will find all checkpoints)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all found checkpoints"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed stats for all layers (even valid ones)"
    )
    parser.add_argument(
        "--check_sharing",
        action="store_true",
        help="Check if policy weights are shared across agents"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.checkpoint and not args.log_dir:
        print("[ERROR] Must provide either --checkpoint or --log_dir")
        parser.print_help()
        return
    
    # Single checkpoint inspection
    if args.checkpoint:
        result = inspect_checkpoint(args.checkpoint, verbose=args.verbose, check_sharing=args.check_sharing)
        
        if result:
            # Print final summary
            print(f"\n{'='*80}")
            print(f"FINAL SUMMARY")
            print(f"{'='*80}")
            
            if result.get("sharing_analysis"):
                sharing = result["sharing_analysis"]
                if sharing.get("shared_policy"):
                    print(f"✅ Policy Sharing: YES ({sharing['shared_percentage']:.1f}% weights shared)")
                elif sharing.get("shared_policy") is False:
                    print(f"❌ Policy Sharing: NO (separate policies per agent)")
                else:
                    print(f"⚠️  Policy Sharing: PARTIAL ({sharing['shared_percentage']:.1f}% weights shared)")
            
            if not (result["has_any_nan"] or result["has_any_inf"]):
                print(f"✅ Weight Validity: VALID (no NaN/Inf)")
                print(f"\n✅ Checkpoint is ready to use!")
            else:
                print(f"❌ Weight Validity: CORRUPTED (contains NaN/Inf)")
                print(f"\n❌ Do NOT use this checkpoint for inference!")
            
            print(f"{'='*80}")
    
    # Log directory inspection
    elif args.log_dir:
        checkpoint_paths = find_checkpoints(args.log_dir)
        
        if not checkpoint_paths:
            print(f"[ERROR] No checkpoints found in {args.log_dir}")
            return
        
        print(f"[INFO] Found {len(checkpoint_paths)} checkpoint(s):")
        for i, path in enumerate(checkpoint_paths, 1):
            print(f"  {i}. {path}")
        
        if args.compare and len(checkpoint_paths) > 1:
            compare_checkpoints(checkpoint_paths)
        else:
            # Inspect each checkpoint individually
            for checkpoint_path in checkpoint_paths:
                inspect_checkpoint(checkpoint_path, verbose=args.verbose, check_sharing=args.check_sharing)


if __name__ == "__main__":
    main()