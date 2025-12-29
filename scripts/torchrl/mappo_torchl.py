import torch
import time
import os
from typing import Dict, List, Optional
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn

class MAPPOPolicy(nn.Module):
    """Shared policy network for MAPPO (per-agent observations)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=[256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        self.net = nn.Sequential(*layers)
        
        # Mean and log_std for Gaussian policy
        self.mean_layer = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor):
        """Forward pass.
        
        Args:
            obs: [batch_size, obs_dim]
        
        Returns:
            mean: [batch_size, action_dim]
            std: [batch_size, action_dim]
        """
        features = self.net(obs)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std.expand_as(mean)


class CentralizedCritic(nn.Module):
    """Centralized critic for MAPPO (sees all agents' observations)."""
    
    def __init__(self, state_dim: int, hidden_sizes=[512, 512, 256]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor):
        """Forward pass.
        
        Args:
            state: [batch_size, state_dim] (centralized state)
        
        Returns:
            value: [batch_size, 1]
        """
        return self.net(state)


def print_tensordict_shapes(td: TensorDict, name: str = "TensorDict"):
    """
    Prints all keys in a TensorDict with their respective shapes.
    
    Args:
        td: The TensorDict to inspect
        name: A descriptive name for the TensorDict (for the header)
    """
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"Batch Size: {td.batch_size}")
    print(f"Device: {td.device}")
    print(f"\nKeys and Shapes:")
    print(f"{'-'*70}")
    
    # Get all keys (nested and leaf nodes)
    all_keys = td.keys(include_nested=True, leaves_only=True)
    
    # Convert all keys to strings for sorting
    key_strings = []
    for key in all_keys:
        if isinstance(key, tuple):
            key_str = " -> ".join(str(k) for k in key)
        else:
            key_str = str(key)
        key_strings.append((key_str, key))
    
    # Sort by string representation
    key_strings.sort(key=lambda x: x[0])
    
    # Print sorted keys with shapes
    for key_str, key in key_strings:
        try:
            tensor = td[key]
            print(f"  {key_str:.<50} {str(tensor.shape)}")
        except Exception as e:
            print(f"  {key_str:.<50} ERROR: {e}")
    
    print(f"{'='*70}\n")


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization implementation using TorchRL.
    """
    def __init__(
        self,
        env,
        policy,
        critic,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,  # Value loss coefficient
        c2: float = 0.01,  # Entropy coefficient
        n_epochs: int = 10,
        batch_size: int = 64,
        n_agents: int = 5,
        frames_per_batch: int = 1000,
        model_name: str = "mappo_run",
        experiment_name: Optional[str] = None,  # ✅ Added experiment_name parameter
        log_dir: Optional[str] = None,  # ✅ Allow external log directory
        checkpoint_interval: int = 100
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.env = env
        self.policy = policy
        self.n_agents = n_agents
        self.critic = critic
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name
        self.experiment_name = experiment_name or "default_experiment"  # ✅ Use provided name

        # ✅ Use external log_dir if provided (from mappo_train.py)
        if log_dir is not None:
            self.run_dir = log_dir
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
            self.log_dir = os.path.join(self.run_dir, "logs")
            
            # Create subdirectories
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"[INFO] Using external log directory")
            print(f"{'='*80}")
            print(f"Experiment: {self.experiment_name}")
            print(f"Model name: {self.model_name}")
            print(f"Run directory: {self.run_dir}")
            print(f"Checkpoint directory: {self.checkpoint_dir}")
            print(f"TensorBoard log directory: {self.log_dir}")
            print(f"{'='*80}\n")
        else:
            # ✅ Fallback: Create directories using experiment_name from config
            time_str = time.strftime('%Y%m%d-%H%M%S')
            
            # Use experiment_name from config as base directory
            base_dir = self.experiment_name
            training_dir = os.path.join(base_dir, "training")
            
            # Create training directory if it doesn't exist
            if not os.path.exists(training_dir):
                os.makedirs(training_dir, exist_ok=True)
                print(f"Created training directory: {training_dir}")
            
            # Create run-specific directories
            self.run_dir = os.path.join(training_dir, "runs", f"{self.model_name}_{time_str}")
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
            self.log_dir = os.path.join(self.run_dir, "logs")
            
            # Create all subdirectories
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"[INFO] Created new experiment directories")
            print(f"{'='*80}")
            print(f"Experiment: {self.experiment_name}")
            print(f"Model name: {self.model_name}")
            print(f"Run directory: {self.run_dir}")
            print(f"Checkpoint directory: {self.checkpoint_dir}")
            print(f"TensorBoard log directory: {self.log_dir}")
            print(f"{'='*80}\n")
        
        # Data collector
        self.collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
            reset_at_each_iter=True,
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=batch_size
        )
        
        # PPO loss
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=True,
            loss_critic_type="l2",
            entropy_coeff=c2,
            critic_coeff=c1,
            normalize_advantage=False,
            safe=True  # Ignore "next" keys automatically
        )
        
        # Setting GAE
        self.loss_module.set_keys(
            reward=env.reward_key,
            action=env.action_key,
            value="state_value",
            done="done",
        )
        self.loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=gae_lambda)
        self.gae = self.loss_module.value_estimator

        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=lr)
        
        # TensorBoard Summary Writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.last_avg_reward = 0.0
        
        print(f"[INFO] MAPPO trainer initialized successfully")
        print(f"[INFO] TensorBoard logs: tensorboard --logdir={self.log_dir}\n")
    
    def train(self, total_frames: int):
        """
        Training loop for MAPPO based on TorchRL tutorial.
        """
        # Initialize the progress bar
        progress_bar = tqdm(total=total_frames, unit="frames", desc="Training")
        
        collected_frames = 0

        for i, tensordict_data in enumerate(self.collector):
            # Stop once total frames reached
            if collected_frames >= total_frames:
                break

            # --- GAE Computation ---
            with torch.no_grad():
                self.gae(tensordict_data)
            
            # Remove collector metadata if present
            if "collector" in tensordict_data.keys():
                del tensordict_data["collector"]
            
            # --- Learning ---
            self.buffer.extend(tensordict_data)

            # Initialize loss tracking for this iteration
            total_loss_objective = 0.0
            total_loss_critic = 0.0
            total_loss_entropy = 0.0

            for _ in range(self.n_epochs):
                for _ in range(self.collector.frames_per_batch // self.batch_size):
                    subdata = self.buffer.sample()
                    
                    # Ensure the sub-batch is on the correct device
                    subdata = subdata.to(self.device)

                    # Flatten the batch for processing
                    subdata.batch_size = torch.Size([self.batch_size, self.n_agents])
                    mini_batch = subdata.reshape(-1)

                    loss_vals = self.loss_module(mini_batch)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Accumulate losses
                    total_loss_objective += loss_vals["loss_objective"].item()
                    total_loss_critic += loss_vals["loss_critic"].item()
                    total_loss_entropy += loss_vals["loss_entropy"].item()

            # Update the policy weights in the collector
            self.collector.update_policy_weights_()

            # --- Checkpointing ---
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(i + 1)

            # --- Logging ---
            frames_in_batch = tensordict_data.numel()
            progress_bar.update(frames_in_batch)
            collected_frames += frames_in_batch
            
            # Calculate average losses
            num_updates = self.n_epochs * (self.collector.frames_per_batch // self.batch_size)
            if num_updates > 0:
                avg_loss_objective = total_loss_objective / num_updates
                avg_loss_critic = total_loss_critic / num_updates
                avg_loss_entropy = total_loss_entropy / num_updates
            else:
                avg_loss_objective = 0.0
                avg_loss_critic = 0.0
                avg_loss_entropy = 0.0

            avg_total_loss = avg_loss_objective + avg_loss_critic + avg_loss_entropy
            
            # Update average reward
            self.last_avg_reward = tensordict_data["next", "reward"].mean().item()

            # TensorBoard logging
            self.writer.add_scalar("Loss/Total", avg_total_loss, collected_frames)
            self.writer.add_scalar("Loss/Policy", avg_loss_objective, collected_frames)
            self.writer.add_scalar("Loss/Value", avg_loss_critic, collected_frames)
            self.writer.add_scalar("Loss/Entropy", avg_loss_entropy, collected_frames)
            self.writer.add_scalar("Reward/Average", self.last_avg_reward, collected_frames)

            # Update progress bar
            progress_bar.set_postfix({
                "Avg Reward": f"{self.last_avg_reward:.2f}",
                "Total Loss": f"{avg_total_loss:.4f}",
                "Policy Loss": f"{avg_loss_objective:.4f}",
                "Value Loss": f"{avg_loss_critic:.4f}",
            })
        
        # Close progress bar
        progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint(collected_frames)
        print(f"\n[INFO] Training complete! Final checkpoint saved.")
        print(f"[INFO] TensorBoard logs: tensorboard --logdir={self.log_dir}")

    def save_checkpoint(self, iteration: int):
        """Saves a checkpoint of the policy and critic models."""
        policy_filename = f'policy_{self.model_name}_iter_{iteration}.pth'
        critic_filename = f'critic_{self.model_name}_iter_{iteration}.pth'
        
        policy_path = os.path.join(self.checkpoint_dir, policy_filename)
        critic_path = os.path.join(self.checkpoint_dir, critic_filename)
        
        # Save state dicts
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"\n[Checkpoint] Saved models at iteration {iteration}")
        print(f"  Policy: {policy_path}")
        print(f"  Critic: {critic_path}")

    def update(self, batch: TensorDict) -> Dict[str, float]:
        """Perform a single MAPPO update step on a mini-batch of data."""
        # Ensure the batch is on the correct device
        batch = batch.to(self.device)
        
        # Compute PPO loss
        loss_vals = self.loss_module(batch)
        
        # Total loss
        loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Return detached loss values
        return {key: val.item() for key, val in loss_vals.items()}