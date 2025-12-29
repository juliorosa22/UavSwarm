"""
TorchRL wrapper for Isaac Lab DirectMARLEnv.
Converts Isaac Lab's multi-agent dict format to TorchRL's TensorDict format.

Updated to use TorchRL's new API:
- Composite instead of CompositeSpec
- Unbounded instead of UnboundedContinuousTensorSpec
"""

import torch
from typing import Dict, Tuple, Any, Optional
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    Unbounded,
    Bounded,
    Categorical,
)
import gymnasium as gym
import numpy as np

from isaaclab.envs import DirectMARLEnv


class IsaacLabTorchRLWrapper(EnvBase):
    """Wrapper to make Isaac Lab DirectMARLEnv compatible with TorchRL.
    
    This wrapper:
    1. Converts dict observations/actions to TensorDict format
    2. Handles multi-agent structure for MAPPO
    3. Provides proper specs for TorchRL modules
    4. Properly handles parallel environments (num_envs dimension)
    """
    
    def __init__(
        self,
        env: gym.Env,
        device: str = "cuda:0",
        centralized_critic: bool = True,
    ):
        """Initialize TorchRL wrapper.
        
        Args:
            env: Isaac Lab environment (already created via gym.make)
            device: Device to run on
            centralized_critic: Whether to provide centralized state for critic
        """
        # ✅ Get unwrapped environment first
        self.env = env
        self.unwrapped_env = env.unwrapped
        
        # ✅ Access num_envs from unwrapped environment
        # This is the number of PARALLEL environments (e.g., 256)
        num_envs = self.unwrapped_env.num_envs
        
        # Initialize parent class with correct batch_size
        # batch_size represents the number of parallel environments
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        
        self.centralized_critic = centralized_critic
        
        # Validate that this is a multi-agent environment
        if not isinstance(self.unwrapped_env, DirectMARLEnv):
            raise TypeError(
                f"Environment must be DirectMARLEnv, got {type(self.unwrapped_env)}"
            )
        
        # Get agent information
        self.possible_agents = self.unwrapped_env.possible_agents
        self.num_agents = len(self.possible_agents)
        
        print(f"\n{'='*80}")
        print(f"[INFO] Initializing TorchRL Wrapper for Isaac Lab")
        print(f"{'='*80}")
        print(f"Environment: {type(self.unwrapped_env).__name__}")
        print(f"Number of agents per env: {self.num_agents}")
        print(f"Agents: {self.possible_agents}")
        print(f"Number of parallel environments: {num_envs}")
        print(f"Total agents (num_envs × num_agents): {num_envs * self.num_agents}")
        print(f"Device: {device}")
        print(f"Centralized critic: {centralized_critic}")
        
        # Get observation and action spaces from environment config
        self.obs_spaces = {}
        self.action_spaces = {}
        
        for agent_id in self.possible_agents:
            # ✅ Access from DirectMARLEnv's observation_spaces and action_spaces dicts
            obs_space = self.unwrapped_env.observation_spaces[agent_id]
            act_space = self.unwrapped_env.action_spaces[agent_id]
            
            self.obs_spaces[agent_id] = obs_space
            self.action_spaces[agent_id] = act_space
            
            print(f"\n{agent_id}:")
            print(f"  Observation space: {obs_space.shape} (per agent)")
            print(f"  Action space: {act_space.shape} (per agent)")
            if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                print(f"  Action bounds: [{act_space.low[0]:.2f}, {act_space.high[0]:.2f}]")
        
        # Build TorchRL specs
        self._make_specs()
        
        print(f"\n✅ TorchRL wrapper initialized successfully")
        print(f"{'='*80}\n")
    
    def _make_specs(self):
        """Create TorchRL specs for observations, actions, rewards, etc.
        
        Dimensions:
        - Observations: [num_envs, obs_dim] per agent (e.g., [256, 23])
        - Actions: [num_envs, action_dim] per agent (e.g., [256, 4])
        - State (centralized): [num_envs, state_dim] (e.g., [256, 115] where 115 = 5 agents × 23)
        - Rewards: [num_envs, 1] per agent
        - Done: [num_envs, 1] (shared)
        """
        
        # ✅ Observation specs: [num_envs, obs_dim] per agent
        obs_specs = {}
        for agent_id in self.possible_agents:
            obs_shape = self.obs_spaces[agent_id].shape  # (23,)
            # Full shape: [num_envs, 23]
            obs_specs[agent_id] = Unbounded(
                shape=(*self.batch_size, *obs_shape),
                device=self.device,
                dtype=torch.float32,
            )
        
        self.observation_spec = Composite(
            observation=Composite(obs_specs, shape=self.batch_size),
            shape=self.batch_size,
        )
        
        # ✅ Action specs: [num_envs, action_dim] per agent
        action_specs = {}
        for agent_id in self.possible_agents:
            act_space = self.action_spaces[agent_id]
            act_shape = act_space.shape  # (4,)
            
            if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                # ✅ FIXED: Use scalar bounds if all dimensions have same bounds
                low = act_space.low   # (4,)
                high = act_space.high # (4,)
                
                # Check if all bounds are the same (common case for continuous control)
                if np.allclose(low, low[0]) and np.allclose(high, high[0]):
                    # Use scalar bounds (much simpler and works with any shape)
                    action_specs[agent_id] = Bounded(
                        low=float(low[0]),
                        high=float(high[0]),
                        shape=(*self.batch_size, *act_shape),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    # Different bounds per dimension - need full tensor
                    # Create bounds that match the full shape
                    full_shape = (*self.batch_size, *act_shape)
                    
                    # Expand low/high to full shape
                    low_tensor = torch.tensor(low, dtype=torch.float32, device=self.device)
                    high_tensor = torch.tensor(high, dtype=torch.float32, device=self.device)
                    
                    # Expand: (4,) -> (1, 4) -> (num_envs, 4)
                    low_expanded = low_tensor.unsqueeze(0).expand(full_shape)
                    high_expanded = high_tensor.unsqueeze(0).expand(full_shape)
                    
                    action_specs[agent_id] = Bounded(
                        low=low_expanded,
                        high=high_expanded,
                        shape=full_shape,
                        device=self.device,
                        dtype=torch.float32,
                    )
            else:
                # Unbounded actions
                action_specs[agent_id] = Unbounded(
                    shape=(*self.batch_size, *act_shape),
                    device=self.device,
                    dtype=torch.float32,
                )
        
        self.action_spec = Composite(
            action=Composite(action_specs, shape=self.batch_size),
            shape=self.batch_size,
        )
        
        # ✅ Reward specs: [num_envs, 1] per agent
        reward_specs = {}
        for agent_id in self.possible_agents:
            reward_specs[agent_id] = Unbounded(
                shape=(*self.batch_size, 1),
                device=self.device,
                dtype=torch.float32,
            )
        
        self.reward_spec = Composite(
            reward=Composite(reward_specs, shape=self.batch_size),
            shape=self.batch_size,
        )
        
        # ✅ Done spec: [num_envs, 1] (shared across all agents)
        self.done_spec = Composite(
            done=Categorical(
                n=2,  # Binary: done or not done
                shape=(*self.batch_size, 1),
                device=self.device,
                dtype=torch.bool,
            ),
            terminated=Categorical(
                n=2,
                shape=(*self.batch_size, 1),
                device=self.device,
                dtype=torch.bool,
            ),
            truncated=Categorical(
                n=2,
                shape=(*self.batch_size, 1),
                device=self.device,
                dtype=torch.bool,
            ),
            shape=self.batch_size,
        )
        
        # ✅ Centralized state spec (for MAPPO critic): [num_envs, state_dim]
        if self.centralized_critic:
            # Get centralized state from environment
            # This concatenates all agent observations: [num_envs, num_agents * obs_dim]
            dummy_state = self.unwrapped_env._get_states(dummy=True)
            state_dim = dummy_state.shape[1]  # Should be num_agents * single_observation_space
            
            # Expected: 5 agents × 23 obs = 115
            expected_state_dim = self.num_agents * self.obs_spaces[self.possible_agents[0]].shape[0]
            
            if state_dim != expected_state_dim:
                print(f"⚠️  WARNING: State dimension mismatch!")
                print(f"   Expected: {expected_state_dim} (num_agents={self.num_agents} × obs_dim={self.obs_spaces[self.possible_agents[0]].shape[0]})")
                print(f"   Got: {state_dim}")
            
            self.state_spec = Composite(
                state=Unbounded(
                    shape=(*self.batch_size, state_dim),
                    device=self.device,
                    dtype=torch.float32,
                ),
                shape=self.batch_size,
            )
            
            print(f"\n[INFO] Centralized State Dimensions:")
            print(f"  State dimension: {state_dim}")
            print(f"  Expected (num_agents × obs_dim): {expected_state_dim}")
            print(f"  State shape: {(*self.batch_size, state_dim)}")
    
    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """Reset environment and return initial observations as TensorDict.
        
        Returns:
            TensorDict with:
            - observation: Dict[agent_id -> Tensor[num_envs, obs_dim]]
            - state: Tensor[num_envs, state_dim] (if centralized_critic=True)
            - done: Tensor[num_envs, 1]
            - terminated: Tensor[num_envs, 1]
            - truncated: Tensor[num_envs, 1]
        """
        # Call Isaac Lab reset
        obs_dict, info_dict = self.env.reset()
        
        # Convert to TensorDict format
        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        # ✅ Add observations: Dict[agent_id -> Tensor[num_envs, obs_dim]]
        obs_td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for agent_id in self.possible_agents:
            obs = obs_dict[agent_id]  # Shape: [num_envs, obs_dim]
            obs_td[agent_id] = obs.to(self.device)
        td["observation"] = obs_td
        
        # ✅ Add centralized state: Tensor[num_envs, state_dim]
        if self.centralized_critic:
            state = self.unwrapped_env._get_states()  # Shape: [num_envs, state_dim]
            td["state"] = state.to(self.device)
        
        # ✅ Initialize done flags: Tensor[num_envs, 1]
        td["done"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["terminated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["truncated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        
        return td
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Step environment with actions from TensorDict.
        
        Args:
            tensordict: TensorDict with actions[agent_id] -> Tensor[num_envs, action_dim]
        
        Returns:
            TensorDict with:
            - observation: Dict[agent_id -> Tensor[num_envs, obs_dim]]
            - reward: Dict[agent_id -> Tensor[num_envs, 1]]
            - state: Tensor[num_envs, state_dim] (if centralized_critic=True)
            - done: Tensor[num_envs, 1]
            - terminated: Tensor[num_envs, 1]
            - truncated: Tensor[num_envs, 1]
            - next: Nested TensorDict with next observations/state
        """
        # ✅ Extract actions from TensorDict: Dict[agent_id -> Tensor[num_envs, action_dim]]
        actions_dict = {}
        for agent_id in self.possible_agents:
            actions_dict[agent_id] = tensordict["action"][agent_id]
        
        # Step Isaac Lab environment
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(actions_dict)
        
        # Convert to TensorDict
        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        # ✅ Add observations: Dict[agent_id -> Tensor[num_envs, obs_dim]]
        obs_td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for agent_id in self.possible_agents:
            obs = obs_dict[agent_id]  # Shape: [num_envs, obs_dim]
            obs_td[agent_id] = obs.to(self.device)
        td["observation"] = obs_td
        
        # ✅ Add rewards: Dict[agent_id -> Tensor[num_envs, 1]]
        reward_td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for agent_id in self.possible_agents:
            reward = rewards_dict[agent_id].to(self.device)  # Shape: [num_envs] or [num_envs, 1]
            # Ensure rewards are 2D: [num_envs, 1]
            if reward.ndim == 1:
                reward = reward.unsqueeze(-1)
            reward_td[agent_id] = reward
        td["reward"] = reward_td
        
        # ✅ Add centralized state: Tensor[num_envs, state_dim]
        if self.centralized_critic:
            state = self.unwrapped_env._get_states()  # Shape: [num_envs, state_dim]
            td["state"] = state.to(self.device)
        
        # ✅ Aggregate done flags: Tensor[num_envs, 1]
        # Environment is done if ANY agent is done
        done = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        truncated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        
        for agent_id in self.possible_agents:
            # terminated_dict[agent_id] shape: [num_envs]
            term = terminated_dict[agent_id].to(self.device)
            trunc = truncated_dict[agent_id].to(self.device)
            
            # Expand to [num_envs, 1] for aggregation
            if term.ndim == 1:
                term = term.unsqueeze(-1)
            if trunc.ndim == 1:
                trunc = trunc.unsqueeze(-1)
            
            done = done | term | trunc
            terminated = terminated | term
            truncated = truncated | trunc
        
        td["done"] = done
        td["terminated"] = terminated
        td["truncated"] = truncated
        
        # ✅ Copy next observations and state for advantage calculation
        td["next"] = td.select("observation", "state") if self.centralized_critic else td.select("observation")
        
        return td
    
    def _set_seed(self, seed: Optional[int]):
        """Set random seed."""
        if seed is not None:
            torch.manual_seed(seed)
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def make_torchrl_env(task: str, num_envs: int, device: str = "cuda:0", **env_kwargs):
    """Create Isaac Lab environment wrapped for TorchRL.
    
    Args:
        task: Task name (e.g., "Isaac-UAV-Swarm-Direct-v0")
        num_envs: Number of parallel environments
        device: Device to run on
        **env_kwargs: Additional environment configuration
    
    Returns:
        IsaacLabTorchRLWrapper: TorchRL-compatible environment
    """
    import gymnasium as gym
    
    # Create Isaac Lab environment via gym
    env = gym.make(task, num_envs=num_envs, **env_kwargs)
    
    # Wrap for TorchRL
    torchrl_env = IsaacLabTorchRLWrapper(env, device=device)
    
    return torchrl_env