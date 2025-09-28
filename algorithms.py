"""
algorithms.py - Advanced Reinforcement Learning Algorithms for Sustainable Tourism
Academic implementation for Information Technology & Tourism journal submission
Authors: [Your Name]
Institution: [Your Institution]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque, namedtuple
import random
from abc import ABC, abstractmethod
import copy
import math
from scipy.optimize import linprog
import cvxpy as cp


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done', 'info'])


@dataclass
class AlgorithmConfig:
    """Configuration for reinforcement learning algorithms"""
    learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.001
    batch_size: int = 256
    buffer_size: int = 1000000
    gradient_clip: float = 10.0
    use_mixed_precision: bool = True
    distributed: bool = False
    gradient_accumulation_steps: int = 1
    
    # PPO specific
    ppo_epochs: int = 10
    ppo_clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    
    # MADDPG specific
    num_agents: int = 10
    update_interval: int = 2
    soft_update_interval: int = 100
    
    # Hierarchical RL specific
    manager_update_freq: int = 10
    worker_update_freq: int = 1
    subgoal_horizon: int = 10
    
    # Constraint optimization
    constraint_violation_penalty: float = 100.0
    lagrange_multiplier_lr: float = 1e-3
    constraint_tolerance: float = 1e-3


class ReplayBuffer:
    """Experience replay buffer with prioritization support"""
    
    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, *args, priority: Optional[float] = None):
        """Add experience to buffer"""
        experience = Experience(*args)
        self.buffer.append(experience)
        
        if self.prioritized:
            max_priority = max(self.priorities) if self.priorities else 1.0
            priority = priority if priority is not None else max_priority
            self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch from buffer with optional prioritization"""
        if self.prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            experiences = [self.buffer[idx] for idx in indices]
            
            # Importance sampling weights
            total = len(self.buffer)
            weights = (total * probabilities[indices]) ** (-beta)
            weights /= weights.max()
            
            return experiences, indices, weights
        else:
            # Uniform sampling
            experiences = random.sample(self.buffer, batch_size)
            return experiences, None, np.ones(batch_size)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for prioritized replay"""
        if self.prioritized and indices is not None:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)


class BaseAlgorithm(ABC):
    """Abstract base class for RL algorithms"""
    
    def __init__(self, model: nn.Module, config: AlgorithmConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Gradient accumulation
        self.accumulation_steps = config.gradient_accumulation_steps
        self.accumulated_steps = 0
        
        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'episodes': 0,
            'average_reward': 0,
            'average_loss': 0
        }
    
    @abstractmethod
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for the algorithm"""
        pass
    
    @abstractmethod
    def select_actions(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select actions given states"""
        pass
    
    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """Execute one training step"""
        pass
    
    @abstractmethod
    def store_transition(self, *args):
        """Store transition in memory"""
        pass
    
    def sync_gradients(self):
        """Synchronize gradients across distributed processes"""
        if self.config.distributed:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()
    
    def clip_gradients(self):
        """Clip gradients to prevent explosion"""
        if self.config.gradient_clip > 0:
            clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)


class MADDPG(BaseAlgorithm):
    """Multi-Agent Deep Deterministic Policy Gradient
    
    Implements centralized training with decentralized execution for multi-agent coordination.
    Each agent has its own actor network while sharing a centralized critic during training.
    """
    
    def __init__(self, model: nn.Module, config: AlgorithmConfig):
        # Initialize actors and critics for each agent
        self.num_agents = config.num_agents
        self.actors = nn.ModuleList([copy.deepcopy(model) for _ in range(self.num_agents)])
        self.critics = nn.ModuleList([self._create_critic(model) for _ in range(self.num_agents)])
        
        # Target networks
        self.target_actors = nn.ModuleList([copy.deepcopy(actor) for actor in self.actors])
        self.target_critics = nn.ModuleList([copy.deepcopy(critic) for critic in self.critics])
        
        # Freeze target networks
        for target_actor, target_critic in zip(self.target_actors, self.target_critics):
            for param in target_actor.parameters():
                param.requires_grad = False
            for param in target_critic.parameters():
                param.requires_grad = False
        
        # Create module containing all networks for optimizer
        self.all_networks = nn.ModuleList(self.actors + self.critics)
        
        super().__init__(self.all_networks, config)
        
        # Memory buffer for each agent
        self.memory = [ReplayBuffer(config.buffer_size, prioritized=True) 
                      for _ in range(self.num_agents)]
        
        # Noise process for exploration
        self.noise_processes = [OrnsteinUhlenbeckProcess(
            size=self._get_action_dim(model), theta=0.15, sigma=0.2
        ) for _ in range(self.num_agents)]
        
        self.update_counter = 0
    
    def _create_critic(self, actor_model: nn.Module) -> nn.Module:
        """Create centralized critic network"""
        state_dim = self._get_state_dim(actor_model)
        action_dim = self._get_action_dim(actor_model)
        
        # Critic takes all agents' states and actions as input
        input_dim = (state_dim + action_dim) * self.num_agents
        
        critic = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        return critic.to(self.device)
    
    def _get_state_dim(self, model: nn.Module) -> int:
        """Extract state dimension from model"""
        # This is model-specific; adjust based on actual model architecture
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 256  # Default
    
    def _get_action_dim(self, model: nn.Module) -> int:
        """Extract action dimension from model"""
        # This is model-specific; adjust based on actual model architecture
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        return last_linear.out_features
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizers for actors and critics"""
        actor_params = []
        critic_params = []
        
        for actor, critic in zip(self.actors, self.critics):
            actor_params.extend(actor.parameters())
            critic_params.extend(critic.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=self.config.learning_rate)
        self.critic_optimizer = optim.Adam(critic_params, lr=self.config.critic_learning_rate)
        
        return self.actor_optimizer  # Return primary optimizer
    
    def select_actions(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select actions for all agents"""
        actions = []
        
        for agent_idx, actor in enumerate(self.actors):
            agent_state = states[agent_idx] if states.dim() > 1 else states
            
            with torch.no_grad():
                actor.eval()
                if hasattr(actor, 'forward'):
                    action = actor(agent_state.unsqueeze(0))
                    if isinstance(action, dict):
                        action = action.get('action', action.get('next_poi_logits'))
                else:
                    action = actor(agent_state.unsqueeze(0))
                
                if not evaluate:
                    # Add exploration noise
                    noise = torch.tensor(
                        self.noise_processes[agent_idx].sample(), 
                        device=self.device
                    )
                    action = action + noise
                
                actions.append(action.squeeze(0))
        
        return torch.stack(actions)
    
    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, 
                        rewards: torch.Tensor, next_states: torch.Tensor, 
                        dones: torch.Tensor, infos: Optional[Dict] = None):
        """Store transitions for all agents"""
        for agent_idx in range(self.num_agents):
            self.memory[agent_idx].push(
                states[agent_idx].cpu().numpy() if torch.is_tensor(states) else states[agent_idx],
                actions[agent_idx].cpu().numpy() if torch.is_tensor(actions) else actions[agent_idx],
                rewards[agent_idx].item() if torch.is_tensor(rewards) else rewards[agent_idx],
                next_states[agent_idx].cpu().numpy() if torch.is_tensor(next_states) else next_states[agent_idx],
                dones[agent_idx].item() if torch.is_tensor(dones) else dones[agent_idx],
                infos[agent_idx] if infos else {}
            )
    
    def train_step(self) -> Dict[str, float]:
        """Execute one training step for all agents"""
        if any(len(memory) < self.config.batch_size for memory in self.memory):
            return {'loss': 0.0}
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for agent_idx in range(self.num_agents):
            # Sample from replay buffer
            experiences, indices, weights = self.memory[agent_idx].sample(
                self.config.batch_size
            )
            
            # Convert to tensors
            states = torch.tensor([e.state for e in experiences], device=self.device)
            actions = torch.tensor([e.action for e in experiences], device=self.device)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device)
            next_states = torch.tensor([e.next_state for e in experiences], device=self.device)
            dones = torch.tensor([e.done for e in experiences], device=self.device)
            weights = torch.tensor(weights, device=self.device)
            
            # Get all agents' actions for centralized critic
            all_actions = []
            all_next_actions = []
            
            for i in range(self.num_agents):
                if i == agent_idx:
                    all_actions.append(actions)
                    with torch.no_grad():
                        next_action = self.target_actors[i](next_states)
                        if isinstance(next_action, dict):
                            next_action = next_action.get('action', next_action.get('next_poi_logits'))
                    all_next_actions.append(next_action)
                else:
                    # Sample other agents' actions from their memories
                    other_experiences, _, _ = self.memory[i].sample(self.config.batch_size)
                    other_actions = torch.tensor([e.action for e in other_experiences], device=self.device)
                    other_next_states = torch.tensor([e.next_state for e in other_experiences], device=self.device)
                    
                    all_actions.append(other_actions)
                    with torch.no_grad():
                        other_next_action = self.target_actors[i](other_next_states)
                        if isinstance(other_next_action, dict):
                            other_next_action = other_next_action.get('action', other_next_action.get('next_poi_logits'))
                    all_next_actions.append(other_next_action)
            
            # Concatenate all states and actions for centralized critic
            all_states_actions = torch.cat([states] + all_actions, dim=1)
            all_next_states_actions = torch.cat([next_states] + all_next_actions, dim=1)
            
            # Critic loss
            with torch.no_grad():
                target_q = self.target_critics[agent_idx](all_next_states_actions)
                target_value = rewards.unsqueeze(1) + self.config.gamma * target_q * (1 - dones.unsqueeze(1))
            
            current_q = self.critics[agent_idx](all_states_actions)
            critic_loss = F.mse_loss(current_q, target_value, reduction='none')
            critic_loss = (critic_loss * weights.unsqueeze(1)).mean()
            
            # Actor loss
            actor_action = self.actors[agent_idx](states)
            if isinstance(actor_action, dict):
                actor_action = actor_action.get('action', actor_action.get('next_poi_logits'))
            
            all_actions_for_actor = all_actions.copy()
            all_actions_for_actor[agent_idx] = actor_action
            actor_input = torch.cat([states] + all_actions_for_actor, dim=1)
            
            actor_loss = -self.critics[agent_idx](actor_input).mean()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            if self.config.use_mixed_precision:
                self.scaler.scale(critic_loss).backward()
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                critic_loss.backward()
                self.clip_gradients()
                self.critic_optimizer.step()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            if self.config.use_mixed_precision:
                self.scaler.scale(actor_loss).backward()
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                actor_loss.backward()
                self.clip_gradients()
                self.actor_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            # Update priorities in replay buffer
            if self.memory[agent_idx].prioritized:
                td_errors = (current_q - target_value).abs().detach().cpu().numpy()
                self.memory[agent_idx].update_priorities(indices, td_errors.squeeze() + 1e-6)
        
        # Soft update target networks
        self.update_counter += 1
        if self.update_counter % self.config.soft_update_interval == 0:
            self._soft_update_targets()
        
        return {
            'actor_loss': total_actor_loss / self.num_agents,
            'critic_loss': total_critic_loss / self.num_agents
        }
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for actor, target_actor, critic, target_critic in zip(
            self.actors, self.target_actors, self.critics, self.target_critics
        ):
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization with specialized advantage estimation
    
    Implements PPO with Generalized Advantage Estimation (GAE) for stable policy updates
    in the sequential decision-making context of tourism itinerary planning.
    """
    
    def __init__(self, model: nn.Module, config: AlgorithmConfig):
        super().__init__(model, config)
        
        # Separate value head if not included in model
        if not hasattr(model, 'value_head'):
            state_dim = self._get_state_dim(model)
            self.value_head = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(self.device)
        else:
            self.value_head = None
        
        # Trajectory buffer
        self.trajectory_buffer = []
        self.current_trajectory = []
    
    def _get_state_dim(self, model: nn.Module) -> int:
        """Extract state dimension from model"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 256
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for policy and value networks"""
        params = list(self.model.parameters())
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
        return optim.Adam(params, lr=self.config.learning_rate)
    
    def select_actions(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select actions using current policy"""
        with torch.no_grad():
            self.model.eval()
            output = self.model(states)
            
            if isinstance(output, dict):
                logits = output.get('next_poi_logits', output.get('action_logits'))
            else:
                logits = output
            
            if evaluate:
                # Deterministic action for evaluation
                actions = logits.argmax(dim=-1)
            else:
                # Sample from policy distribution
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
            
            # Store log probabilities for training
            if not evaluate:
                log_probs = dist.log_prob(actions)
                self.current_trajectory.append({
                    'states': states.cpu(),
                    'actions': actions.cpu(),
                    'log_probs': log_probs.cpu()
                })
        
        return actions
    
    def store_transition(self, states: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, next_states: torch.Tensor,
                        dones: torch.Tensor, infos: Optional[Dict] = None):
        """Store transition in current trajectory"""
        if len(self.current_trajectory) > 0:
            self.current_trajectory[-1]['rewards'] = rewards.cpu()
            self.current_trajectory[-1]['dones'] = dones.cpu()
            self.current_trajectory[-1]['next_states'] = next_states.cpu()
        
        # If episode is done, process trajectory
        if dones.any():
            self._process_trajectory()
    
    def _process_trajectory(self):
        """Process completed trajectory with GAE"""
        if len(self.current_trajectory) == 0:
            return
        
        # Stack trajectory data
        states = torch.stack([t['states'] for t in self.current_trajectory])
        actions = torch.stack([t['actions'] for t in self.current_trajectory])
        log_probs = torch.stack([t['log_probs'] for t in self.current_trajectory])
        rewards = torch.stack([t['rewards'] for t in self.current_trajectory])
        dones = torch.stack([t['dones'] for t in self.current_trajectory])
        
        # Calculate values
        with torch.no_grad():
            if self.value_head is not None:
                values = self.value_head(states.to(self.device)).squeeze().cpu()
                next_values = torch.cat([
                    values[1:],
                    torch.zeros(1)
                ])
            else:
                output = self.model(states.to(self.device))
                values = output['value'].squeeze().cpu()
                next_values = torch.cat([
                    values[1:],
                    torch.zeros(1)
                ])
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(
            rewards, values, next_values, dones
        )
        
        # Calculate returns
        returns = advantages + values
        
        # Store processed trajectory
        self.trajectory_buffer.append({
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns
        })
        
        # Clear current trajectory
        self.current_trajectory = []
    
    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                       next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def train_step(self) -> Dict[str, float]:
        """Execute PPO training step"""
        if len(self.trajectory_buffer) == 0:
            return {'loss': 0.0}
        
        # Combine all trajectories
        all_states = torch.cat([t['states'] for t in self.trajectory_buffer])
        all_actions = torch.cat([t['actions'] for t in self.trajectory_buffer])
        all_log_probs = torch.cat([t['log_probs'] for t in self.trajectory_buffer])
        all_advantages = torch.cat([t['advantages'] for t in self.trajectory_buffer])
        all_returns = torch.cat([t['returns'] for t in self.trajectory_buffer])
        
        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Create random batches
            num_samples = len(all_states)
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = all_states[batch_indices].to(self.device)
                batch_actions = all_actions[batch_indices].to(self.device)
                batch_log_probs = all_log_probs[batch_indices].to(self.device)
                batch_advantages = all_advantages[batch_indices].to(self.device)
                batch_returns = all_returns[batch_indices].to(self.device)
                
                # Forward pass
                output = self.model(batch_states)
                
                if isinstance(output, dict):
                    logits = output.get('next_poi_logits', output.get('action_logits'))
                    if self.value_head is None:
                        values = output['value'].squeeze()
                    else:
                        values = self.value_head(batch_states).squeeze()
                else:
                    logits = output
                    values = self.value_head(batch_states).squeeze()
                
                # Calculate new log probabilities
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_epsilon, 
                                   1 + self.config.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.clip_gradients()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.clip_gradients()
                    self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear trajectory buffer
        self.trajectory_buffer = []
        
        num_updates = self.config.ppo_epochs * (num_samples // self.config.batch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }


class HierarchicalRL(BaseAlgorithm):
    """Hierarchical Reinforcement Learning for multi-day planning
    
    Implements a two-level hierarchy with manager (high-level) and worker (low-level) policies
    for temporal abstraction in tourism itinerary planning.
    """
    
    def __init__(self, model: nn.Module, config: AlgorithmConfig):
        # Assuming model is HierarchicalPolicyNetwork
        super().__init__(model, config)
        
        # Separate optimizers for manager and worker
        self.manager_optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if 'high_level' in n],
            lr=config.learning_rate
        )
        self.worker_optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if 'low_level' in n],
            lr=config.learning_rate * 2
        )
        
        # Memory buffers
        self.manager_memory = ReplayBuffer(config.buffer_size // 10)
        self.worker_memory = ReplayBuffer(config.buffer_size)
        
        # Current subgoal tracking
        self.current_subgoal = None
        self.subgoal_steps = 0
        self.manager_update_counter = 0
        self.worker_update_counter = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer (returns manager optimizer as primary)"""
        return self.manager_optimizer
    
    def select_actions(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select actions using hierarchical policy"""
        with torch.no_grad():
            self.model.eval()
            
            # Manager decision (every subgoal_horizon steps)
            if self.subgoal_steps % self.config.subgoal_horizon == 0:
                global_state = states[0] if states.dim() > 1 else states
                day_index = torch.tensor([self.subgoal_steps // self.config.subgoal_horizon], 
                                        device=self.device)
                
                high_level_output = self.model.forward_high_level(
                    global_state.unsqueeze(0),
                    day_index
                )
                
                if evaluate:
                    self.current_subgoal = high_level_output['zone_logits'].argmax(dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=high_level_output['zone_logits'])
                    self.current_subgoal = dist.sample()
                
                self.subgoal_steps = 0
            
            # Worker decision (every step)
            zone_one_hot = F.one_hot(self.current_subgoal, num_classes=self.model.num_zones).float()
            low_level_output = self.model.forward_low_level(
                states.unsqueeze(0) if states.dim() == 1 else states,
                zone_one_hot
            )
            
            if evaluate:
                actions = low_level_output['poi_logits'].argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=low_level_output['poi_logits'])
                actions = dist.sample()
            
            self.subgoal_steps += 1
            
        return actions.squeeze()
    
    def store_transition(self, states: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, next_states: torch.Tensor,
                        dones: torch.Tensor, infos: Optional[Dict] = None):
        """Store transitions for hierarchical learning"""
        # Store worker transition
        self.worker_memory.push(
            states.cpu().numpy(),
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
            next_states.cpu().numpy(),
            dones.cpu().numpy(),
            {'subgoal': self.current_subgoal.cpu().numpy() if self.current_subgoal is not None else None}
        )
        
        # Store manager transition at subgoal boundaries
        if self.subgoal_steps % self.config.subgoal_horizon == 0 or dones.any():
            # Calculate cumulative reward over subgoal horizon
            cumulative_reward = rewards.sum().item()  # Simplified; should track over horizon
            
            self.manager_memory.push(
                states.cpu().numpy(),
                self.current_subgoal.cpu().numpy() if self.current_subgoal is not None else 0,
                cumulative_reward,
                next_states.cpu().numpy(),
                dones.cpu().numpy(),
                {}
            )
    
    def train_step(self) -> Dict[str, float]:
        """Execute hierarchical training step"""
        total_manager_loss = 0.0
        total_worker_loss = 0.0
        
        # Train manager
        if self.manager_update_counter % self.config.manager_update_freq == 0 and \
           len(self.manager_memory) >= self.config.batch_size:
            
            experiences, _, _ = self.manager_memory.sample(self.config.batch_size)
            
            states = torch.tensor([e.state for e in experiences], device=self.device)
            subgoals = torch.tensor([e.action for e in experiences], device=self.device)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device)
            next_states = torch.tensor([e.next_state for e in experiences], device=self.device)
            dones = torch.tensor([e.done for e in experiences], device=self.device)
            
            # Manager loss (simplified policy gradient)
            day_indices = torch.zeros(len(states), device=self.device).long()
            high_level_output = self.model.forward_high_level(states, day_indices)
            
            dist = torch.distributions.Categorical(logits=high_level_output['zone_logits'])
            log_probs = dist.log_prob(subgoals)
            
            # Calculate advantages (simplified)
            advantages = rewards - high_level_output['subgoal_value'].squeeze()
            
            manager_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(high_level_output['subgoal_value'].squeeze(), rewards)
            
            total_manager_loss = manager_loss + value_loss
            
            self.manager_optimizer.zero_grad()
            total_manager_loss.backward()
            self.clip_gradients()
            self.manager_optimizer.step()
            
            total_manager_loss = total_manager_loss.item()
        
        self.manager_update_counter += 1
        
        # Train worker
        if self.worker_update_counter % self.config.worker_update_freq == 0 and \
           len(self.worker_memory) >= self.config.batch_size:
            
            experiences, _, _ = self.worker_memory.sample(self.config.batch_size)
            
            states = torch.tensor([e.state for e in experiences], device=self.device)
            actions = torch.tensor([e.action for e in experiences], device=self.device)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device)
            next_states = torch.tensor([e.next_state for e in experiences], device=self.device)
            dones = torch.tensor([e.done for e in experiences], device=self.device)
            subgoals = torch.tensor([e.info.get('subgoal', 0) for e in experiences], device=self.device)
            
            # Worker loss
            zone_one_hot = F.one_hot(subgoals, num_classes=self.model.num_zones).float()
            low_level_output = self.model.forward_low_level(states, zone_one_hot)
            
            dist = torch.distributions.Categorical(logits=low_level_output['poi_logits'].squeeze())
            log_probs = dist.log_prob(actions)
            
            # Intrinsic + extrinsic rewards
            intrinsic_rewards = low_level_output['intrinsic_rewards'].squeeze()
            total_rewards = rewards + 0.1 * intrinsic_rewards
            
            worker_loss = -(log_probs * total_rewards.detach()).mean()
            
            self.worker_optimizer.zero_grad()
            worker_loss.backward()
            self.clip_gradients()
            self.worker_optimizer.step()
            
            total_worker_loss = worker_loss.item()
        
        self.worker_update_counter += 1
        
        return {
            'manager_loss': total_manager_loss,
            'worker_loss': total_worker_loss
        }


class ConstrainedOptimization(BaseAlgorithm):
    """Constrained optimization for capacity and sustainability constraints
    
    Implements Lagrangian relaxation and barrier methods to ensure hard constraints
    are satisfied during policy optimization.
    """
    
    def __init__(self, model: nn.Module, capacity_constraints: Dict[int, int], 
                config: AlgorithmConfig):
        super().__init__(model, config)
        
        self.capacity_constraints = capacity_constraints
        
        # Lagrange multipliers for constraints
        self.lagrange_multipliers = nn.ParameterDict({
            str(poi_id): nn.Parameter(torch.zeros(1))
            for poi_id in capacity_constraints.keys()
        })
        
        # Optimizer for Lagrange multipliers
        self.multiplier_optimizer = optim.Adam(
            self.lagrange_multipliers.parameters(),
            lr=config.lagrange_multiplier_lr
        )
        
        # Memory buffer
        self.memory = ReplayBuffer(config.buffer_size)
        
        # Constraint violation tracking
        self.violation_history = deque(maxlen=1000)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for main model"""
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def select_actions(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select actions with constraint awareness"""
        with torch.no_grad():
            self.model.eval()
            output = self.model(states)
            
            if isinstance(output, dict):
                logits = output.get('action_logits', output.get('next_poi_logits'))
            else:
                logits = output
            
            # Apply constraint masking
            masked_logits = self._apply_constraint_mask(logits, states)
            
            if evaluate:
                actions = masked_logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=masked_logits)
                actions = dist.sample()
        
        return actions
    
    def _apply_constraint_mask(self, logits: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Apply masking to ensure constraints are not violated"""
        # This is a simplified version; actual implementation would check current occupancy
        mask = torch.ones_like(logits)
        
        # Mask actions that would violate capacity constraints
        for poi_id, capacity in self.capacity_constraints.items():
            # Check current occupancy (simplified)
            if hasattr(self, 'current_occupancy'):
                if self.current_occupancy.get(poi_id, 0) >= capacity:
                    mask[:, poi_id] = 0
        
        masked_logits = logits + torch.log(mask + 1e-10)
        return masked_logits
    
    def store_transition(self, states: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, next_states: torch.Tensor,
                        dones: torch.Tensor, infos: Optional[Dict] = None):
        """Store transition with constraint violation information"""
        # Calculate constraint violations
        violations = self._calculate_violations(states, actions, infos)
        
        # Augmented info with violations
        augmented_info = infos or {}
        augmented_info['violations'] = violations
        
        self.memory.push(
            states.cpu().numpy(),
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
            next_states.cpu().numpy(),
            dones.cpu().numpy(),
            augmented_info
        )
        
        self.violation_history.append(violations.sum().item())
    
    def _calculate_violations(self, states: torch.Tensor, actions: torch.Tensor,
                            infos: Optional[Dict]) -> torch.Tensor:
        """Calculate constraint violations for given actions"""
        violations = torch.zeros(len(self.capacity_constraints))
        
        if infos is not None:
            for i, (poi_id, capacity) in enumerate(self.capacity_constraints.items()):
                # Check if action violates capacity
                occupancy = infos.get('occupancy', {}).get(poi_id, 0)
                if occupancy > capacity:
                    violations[i] = occupancy - capacity
        
        return violations
    
    def train_step(self) -> Dict[str, float]:
        """Execute constrained optimization training step"""
        if len(self.memory) < self.config.batch_size:
            return {'loss': 0.0}
        
        experiences, _, _ = self.memory.sample(self.config.batch_size)
        
        states = torch.tensor([e.state for e in experiences], device=self.device)
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], device=self.device)
        next_states = torch.tensor([e.next_state for e in experiences], device=self.device)
        dones = torch.tensor([e.done for e in experiences], device=self.device)
        violations = torch.tensor([e.info['violations'] for e in experiences], device=self.device)
        
        # Forward pass
        output = self.model(states)
        if isinstance(output, dict):
            logits = output.get('action_logits', output.get('next_poi_logits'))
            values = output.get('value', torch.zeros(len(states), 1, device=self.device))
        else:
            logits = output
            values = torch.zeros(len(states), 1, device=self.device)
        
        # Policy loss
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        advantages = rewards - values.squeeze()
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Constraint loss with Lagrange multipliers
        constraint_loss = 0
        for i, (poi_id, multiplier) in enumerate(self.lagrange_multipliers.items()):
            violation = violations[:, i].mean()
            constraint_loss += multiplier * violation
        
        # Barrier function for hard constraints
        barrier_loss = self.config.constraint_violation_penalty * violations.sum(dim=1).mean()
        
        # Total loss
        total_loss = policy_loss + constraint_loss + barrier_loss
        
        # Update main model
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.clip_gradients()
        self.optimizer.step()
        
        # Update Lagrange multipliers
        self.multiplier_optimizer.zero_grad()
        (-constraint_loss).backward()
        self.multiplier_optimizer.step()
        
        # Ensure multipliers stay non-negative
        with torch.no_grad():
            for multiplier in self.lagrange_multipliers.values():
                multiplier.clamp_(min=0)
        
        return {
            'policy_loss': policy_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'barrier_loss': barrier_loss.item(),
            'average_violation': violations.mean().item()
        }


class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhlenbeck process for exploration noise"""
    
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2, mu: float = 0.0):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = np.ones(size) * mu
        self.reset()
    
    def reset(self):
        """Reset the process"""
        self.state = np.ones(self.size) * self.mu
    
    def sample(self) -> np.ndarray:
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


def create_algorithm(algorithm_type: str, model: nn.Module, 
                    config: AlgorithmConfig, **kwargs) -> BaseAlgorithm:
    """Factory function for algorithm creation"""
    algorithms = {
        'maddpg': MADDPG,
        'ppo': PPO,
        'hierarchical': HierarchicalRL,
        'constrained': ConstrainedOptimization
    }
    
    if algorithm_type not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_type}")
    
    if algorithm_type == 'constrained':
        return algorithms[algorithm_type](model, kwargs['capacity_constraints'], config)
    else:
        return algorithms[algorithm_type](model, config)