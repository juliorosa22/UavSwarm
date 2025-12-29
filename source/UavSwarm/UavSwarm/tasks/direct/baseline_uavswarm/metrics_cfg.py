from dataclasses import dataclass, field
from typing import Dict
import torch

@dataclass
class EpisodeMetrics:
    """Centralized container for all episodic metrics with automatic logging support.
    
    Attributes:
        Basic metrics (velocities, distances):
            - lin_vel: Linear velocity penalties
            - ang_vel: Angular velocity penalties
            - distance_to_goal: Mean distance to goals
            - collision: Collision penalties
            
        Reward decomposition:
            - mean_reward: Total reward per episode
            - dist_component: Distance-based reward component
            - obs_component: Obstacle avoidance reward component
            - coop_component: Cooperation reward component
            
        Formation metrics (stages 4-5):
            - formation: Formation maintenance penalties
            - swarm_cohesion: Inter-agent distance variance
    """
    # Basic metrics
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor
    distance_to_goal: torch.Tensor
    collision: torch.Tensor
    
    # Reward decomposition
    mean_reward: torch.Tensor
    dist_component: torch.Tensor
    obs_component: torch.Tensor
    coop_component: torch.Tensor
    
    # Formation metrics (optional for stages 4-5)
    formation: torch.Tensor
    swarm_cohesion: torch.Tensor = None  # Optional, initialized later if needed
    
    @classmethod
    def create(cls, num_envs: int, device: str) -> "EpisodeMetrics":
        """Factory method to create zero-initialized metrics.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to allocate tensors on ("cuda" or "cpu")
        
        Returns:
            EpisodeMetrics instance with all buffers initialized to zeros
        """
        return cls(
            lin_vel=torch.zeros(num_envs, dtype=torch.float, device=device),
            ang_vel=torch.zeros(num_envs, dtype=torch.float, device=device),
            distance_to_goal=torch.zeros(num_envs, dtype=torch.float, device=device),
            collision=torch.zeros(num_envs, dtype=torch.float, device=device),
            mean_reward=torch.zeros(num_envs, dtype=torch.float, device=device),
            dist_component=torch.zeros(num_envs, dtype=torch.float, device=device),
            obs_component=torch.zeros(num_envs, dtype=torch.float, device=device),
            coop_component=torch.zeros(num_envs, dtype=torch.float, device=device),
            formation=torch.zeros(num_envs, dtype=torch.float, device=device),
        )
    
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset metrics for specified environments to zero.
        
        Args:
            env_ids: Indices of environments to reset
        """
        self.lin_vel[env_ids] = 0.0
        self.ang_vel[env_ids] = 0.0
        self.distance_to_goal[env_ids] = 0.0
        self.collision[env_ids] = 0.0
        self.mean_reward[env_ids] = 0.0
        self.dist_component[env_ids] = 0.0
        self.obs_component[env_ids] = 0.0
        self.coop_component[env_ids] = 0.0
        self.formation[env_ids] = 0.0
        
        if self.swarm_cohesion is not None:
            self.swarm_cohesion[env_ids] = 0.0
    
    def to_log_dict(
        self, 
        env_ids: torch.Tensor, 
        max_episode_length: float,
        prefix: str = "Episode_Reward"
    ) -> Dict[str, float]:
        """Convert metrics to logging dictionary with normalized values.
        
        Args:
            env_ids: Indices of environments being reset
            max_episode_length: Maximum episode length in seconds (for normalization)
            prefix: Prefix for logging keys (default: "Episode_Reward")
        
        Returns:
            Dictionary mapping metric names to averaged/normalized values
        """
        log_dict = {}
        
        # Basic metrics (averaged across reset environments, normalized by episode length)
        log_dict[f"{prefix}/lin_vel"] = (
            torch.mean(self.lin_vel[env_ids]) / max_episode_length
        ).item()
        
        log_dict[f"{prefix}/ang_vel"] = (
            torch.mean(self.ang_vel[env_ids]) / max_episode_length
        ).item()
        
        log_dict[f"{prefix}/distance_to_goal"] = (
            torch.mean(self.distance_to_goal[env_ids]) / max_episode_length
        ).item()
        
        log_dict[f"{prefix}/collision"] = (
            torch.mean(self.collision[env_ids]) / max_episode_length
        ).item()
        
        # Reward decomposition
        log_dict[f"{prefix}/mean_reward"] = (
            torch.mean(self.mean_reward[env_ids]) / max_episode_length
        ).item()
        
        log_dict[f"{prefix}/dist_component"] = (
            torch.mean(self.dist_component[env_ids]) / max_episode_length
        ).item()
        
        log_dict[f"{prefix}/obs_component"] = (
            torch.mean(self.obs_component[env_ids]) / max_episode_length
        ).item()
        
        log_dict[f"{prefix}/coop_component"] = (
            torch.mean(self.coop_component[env_ids]) / max_episode_length
        ).item()
        
        # Formation metrics
        log_dict[f"{prefix}/formation"] = (
            torch.mean(self.formation[env_ids]) / max_episode_length
        ).item()
        
        if self.swarm_cohesion is not None:
            log_dict[f"{prefix}/swarm_cohesion"] = (
                torch.mean(self.swarm_cohesion[env_ids]) / max_episode_length
            ).item()
        
        return log_dict
    
    def update(
        self,
        lin_vel: torch.Tensor = None,
        ang_vel: torch.Tensor = None,
        distance_to_goal: torch.Tensor = None,
        collision: torch.Tensor = None,
        mean_reward: torch.Tensor = None,
        dist_component: torch.Tensor = None,
        obs_component: torch.Tensor = None,
        coop_component: torch.Tensor = None,
        formation: torch.Tensor = None,
        swarm_cohesion: torch.Tensor = None,
    ) -> None:
        """Accumulate metrics (in-place addition).
        
        Args:
            **kwargs: Metric tensors to add (shape: (num_envs,))
                     Only non-None values are accumulated
        """
        if lin_vel is not None:
            self.lin_vel += lin_vel
        if ang_vel is not None:
            self.ang_vel += ang_vel
        if distance_to_goal is not None:
            self.distance_to_goal += distance_to_goal
        if collision is not None:
            self.collision += collision
        if mean_reward is not None:
            self.mean_reward += mean_reward
        if dist_component is not None:
            self.dist_component += dist_component
        if obs_component is not None:
            self.obs_component += obs_component
        if coop_component is not None:
            self.coop_component += coop_component
        if formation is not None:
            self.formation += formation
        if swarm_cohesion is not None and self.swarm_cohesion is not None:
            self.swarm_cohesion += swarm_cohesion
    
    def get_mean(self, metric_name: str, env_ids: torch.Tensor = None) -> float:
        """Get mean value of a specific metric.
        
        Args:
            metric_name: Name of the metric ("lin_vel", "mean_reward", etc.)
            env_ids: Optional environment indices to average over (default: all envs)
        
        Returns:
            Mean value as Python float
        """
        metric_tensor = getattr(self, metric_name)
        
        if env_ids is None:
            return metric_tensor.mean().item()
        else:
            return metric_tensor[env_ids].mean().item()