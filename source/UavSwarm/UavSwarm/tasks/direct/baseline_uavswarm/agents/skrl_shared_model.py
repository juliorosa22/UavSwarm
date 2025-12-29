import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class SharedPolicyModel(GaussianMixin, Model):
    """Shared policy network for all agents (homogeneous multi-agent).
    
    This policy is trained on per-agent observations and shared across ALL agents,
    making it scalable to any number of agents at test time.
    
    Architecture:
        Input: Per-agent observation (23 dims)
        Hidden: [128, 128]
        Output: Actions (4 dims) + log_std parameter
    """
    
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=True, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, 
                 initial_log_std=0.0, reduction="sum"):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )
        
        # Get input dimension from observation space
        if hasattr(observation_space, 'shape'):
            self.obs_dim = observation_space.shape[0]
        else:
            self.obs_dim = 23  # Fallback
        
        print(f"\n{'='*70}")
        print(f"Shared Policy Network (homogeneous agents)")
        print(f"{'='*70}")
        print(f"  Input: {self.obs_dim} dims (per-agent observations)")
        print(f"  Architecture: {self.obs_dim} → 128 → 128 → {self.num_actions}")
        print(f"  This policy will be SHARED across ALL agents")
        print(f"  ✅ Scalable to any number of agents at test time")
        
        # Policy network
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std),
            requires_grad=True
        )
        
        # Count parameters
        params = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {params:,}")
        print(f"{'='*70}\n")
    
    def compute(self, inputs, role=""):
        """Forward pass for policy.
        
        Args:
            inputs: Dict with "states" key containing per-agent observations
                Shape: (batch_size, obs_dim)
            role: Unused (for compatibility)
        
        Returns:
            Tuple of (mean_actions, log_std, {})
        """
        states = inputs["states"]
        
        # Handle flattened tensorized space if needed
        if hasattr(self.observation_space, '__len__'):
            states = unflatten_tensorized_space(self.observation_space, states)
        
        # Clip observations to prevent extreme values
        states = torch.clamp(states, min=-10.0, max=10.0)
        
        # Forward pass
        features = self.net(states)
        mean_actions = self.mean_layer(features)
        
        return mean_actions, self.log_std_parameter, {}


class CentralizedCriticModel(DeterministicMixin, Model):
    """Centralized critic network for MAPPO (sees all agents)."""
    
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, num_agents=5):
        
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        
        # Get state dimension
        if hasattr(observation_space, 'shape'):
            self.state_dim = observation_space.shape[0]
        else:
            self.state_dim = num_agents * 23
        
        self.num_agents = num_agents
        self.obs_per_agent = self.state_dim // num_agents
        
        print(f"\n{'='*70}")
        print(f"Centralized Critic Network")
        print(f"{'='*70}")
        print(f"  Input: {self.state_dim} dims (all {num_agents} agents' observations)")
        print(f"  Per-agent obs: {self.obs_per_agent} dims")
        print(f"  Architecture: {self.state_dim} → 256 → 256 → 256 → 1")  # ✅ Updated
        print(f"  Provides centralized value estimation for coordination")
        
        # ✅ FIX: WIDER network for centralized critic
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 256),  # ✅ 115 → 256 (was 128)
            nn.ReLU(),
            nn.Linear(256, 256),              # ✅ 256 → 256 (was 128)
            nn.ReLU(),
            nn.Linear(256, 256),              # ✅ 256 → 256 (was 128)
            nn.ReLU(),
        )
        
        self.value_layer = nn.Linear(256, 1)  # ✅ 256 → 1 (was 128)
        
        params = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {params:,}")
        print(f"{'='*70}\n")
    
    def compute(self, inputs, role=""):
        """Forward pass for critic."""
        states = inputs["states"]
        
        # ✅ Add shape debugging
        if states.shape[-1] != self.state_dim:
            print(f"⚠️  WARNING: State shape mismatch!")
            print(f"   Expected: (..., {self.state_dim})")
            print(f"   Got: {states.shape}")
        
        # ✅ Less aggressive clamping
        states = torch.clamp(states, min=-50.0, max=50.0)
        
        # Forward pass
        features = self.net(states)
        value = self.value_layer(features)
        
        return value, {}


class AdaptiveCentralizedCriticModel(DeterministicMixin, Model):
    """Adaptive centralized critic that handles variable number of agents.
    
    This version uses attention mechanism to aggregate agent observations,
    making it truly scalable to any number of agents at test time.
    """
    
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, obs_per_agent=23):
        
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        
        self.obs_per_agent = obs_per_agent
        
        print(f"\n{'='*70}")
        print(f"Adaptive Centralized Critic (Attention-Based)")
        print(f"{'='*70}")
        print(f"  Per-agent obs: {obs_per_agent} dims")
        print(f"  ✅ Truly scalable to ANY number of agents")
        print(f"  Uses attention to aggregate agent observations")
        
        # Per-agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_per_agent, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Attention mechanism for aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        params = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {params:,}")
        print(f"{'='*70}\n")
    
    def compute(self, inputs, role=""):
        """Forward pass with attention aggregation.
        
        Args:
            inputs: Dict with "states" key
                Shape: (batch_size, num_agents * obs_per_agent)
        """
        states = inputs["states"]
        
        # Handle flattened tensorized space
        if hasattr(self.observation_space, '__len__'):
            states = unflatten_tensorized_space(self.observation_space, states)
        
        states = torch.clamp(states, min=-10.0, max=10.0)
        
        batch_size = states.shape[0]
        num_agents = states.shape[1] // self.obs_per_agent
        
        # Reshape to (batch_size, num_agents, obs_per_agent)
        agent_obs = states.reshape(batch_size, num_agents, self.obs_per_agent)
        
        # Encode each agent's observation
        # (batch_size, num_agents, obs_per_agent) -> (batch_size, num_agents, 128)
        agent_features = self.agent_encoder(agent_obs)
        
        # Apply attention (aggregate across agents)
        # Output: (batch_size, num_agents, 128)
        attended_features, _ = self.attention(
            agent_features, agent_features, agent_features
        )
        
        # Global pooling (mean across agents)
        # (batch_size, num_agents, 128) -> (batch_size, 128)
        global_features = attended_features.mean(dim=1)
        
        # Compute value
        value = self.value_net(global_features)
        
        return value, {}