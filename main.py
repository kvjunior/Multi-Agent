"""
main.py - Central Orchestrator for Multi-Agent Reinforcement Learning in Sustainable Tourism
Academic implementation for Information Technology & Tourism journal submission
Authors: [Your Name]
Institution: [Your Institution]
"""

import os
import sys
import argparse
import yaml
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
import wandb
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Import custom modules (to be implemented in other files)
from models import (
    TransformerItineraryModel,
    GraphNeuralPOINetwork,
    HierarchicalPolicyNetwork,
    UncertaintyAwareQNetwork
)
from environments import MultiAgentTourismEnv
from algorithms import MADDPG, PPO, HierarchicalRL, ConstrainedOptimization
from utilities import (
    DataLoader,
    MetricsCalculator,
    Visualizer,
    set_random_seeds,
    get_device,
    save_checkpoint,
    load_checkpoint
)


@dataclass
class ExperimentConfig:
    """Configuration dataclass for experiment parameters with validation"""
    
    # General settings
    experiment_name: str
    seed: int = 42
    device: str = "cuda"
    num_gpus: int = 4
    mixed_precision: bool = True
    
    # Data settings
    data_path: str = "./data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Environment settings
    num_agents: int = 10
    max_episode_steps: int = 100
    poi_capacity_factor: float = 1.0
    crowding_penalty_weight: float = 0.3
    sustainability_weight: float = 0.2
    
    # Model architecture
    model_type: str = "transformer"  # ["transformer", "graph", "hierarchical", "uncertainty"]
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    embedding_dim: int = 128
    
    # Training settings
    algorithm: str = "maddpg"  # ["maddpg", "ppo", "hierarchical", "constrained"]
    batch_size: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.001
    num_episodes: int = 10000
    warmup_episodes: int = 100
    
    # Optimization settings
    use_hyperopt: bool = True
    hyperopt_trials: int = 100
    hyperopt_timeout: int = 3600  # seconds
    
    # Logging settings
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    use_wandb: bool = True
    use_mlflow: bool = True
    use_tensorboard: bool = True
    
    # Checkpoint settings
    checkpoint_dir: str = "./experiments/checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.train_split < 1
        assert 0 < self.val_split < 1
        assert 0 < self.test_split < 1
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6
        assert self.num_agents > 0
        assert self.hidden_dim > 0
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert 0 <= self.gamma <= 1
        assert 0 <= self.tau <= 1


class ExperimentOrchestrator:
    """Main orchestrator for distributed multi-agent RL experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.validate()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize distributed training if using multiple GPUs
        self.distributed = self.config.num_gpus > 1
        self.rank = 0
        self.world_size = 1
        
        # Initialize tracking systems
        self._initialize_tracking()
        
        # Load data
        self.data_loader = DataLoader(self.config.data_path)
        self.train_data, self.val_data, self.test_data = self._prepare_datasets()
        
        # Initialize metrics calculator
        self.metrics = MetricsCalculator()
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Best model tracking
        self.best_val_performance = float('-inf')
        self.patience_counter = 0
        self.early_stopping_patience = 20
        
    def _setup_logging(self) -> None:
        """Configure comprehensive logging system"""
        log_dir = Path("./experiments/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.config.experiment_name}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment configuration: {asdict(self.config)}")
        
    def _initialize_tracking(self) -> None:
        """Initialize experiment tracking systems"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.use_wandb:
            wandb.init(
                project="sustainable-tourism-marl",
                name=f"{self.config.experiment_name}_{timestamp}",
                config=asdict(self.config)
            )
            
        if self.config.use_mlflow:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run(run_name=f"run_{timestamp}")
            mlflow.log_params(asdict(self.config))
            
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=f"./experiments/logs/tensorboard/{self.config.experiment_name}_{timestamp}"
            )
    
    def _prepare_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare and split datasets for training, validation, and testing"""
        self.logger.info("Loading and preparing datasets...")
        
        # Load all data components
        poi_metadata = self.data_loader.load_poi_metadata()
        historical_visits = self.data_loader.load_historical_visits()
        crowding_patterns = self.data_loader.load_crowding_patterns()
        contextual_factors = self.data_loader.load_contextual_factors()
        
        # Merge and preprocess data
        processed_data = self.data_loader.preprocess_data(
            poi_metadata, 
            historical_visits,
            crowding_patterns,
            contextual_factors
        )
        
        # Split data temporally for realistic evaluation
        n_samples = len(processed_data)
        train_idx = int(n_samples * self.config.train_split)
        val_idx = int(n_samples * (self.config.train_split + self.config.val_split))
        
        train_data = processed_data[:train_idx]
        val_data = processed_data[train_idx:val_idx]
        test_data = processed_data[val_idx:]
        
        self.logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def setup_distributed(self, rank: int, world_size: int) -> None:
        """Setup distributed training environment"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        
        torch.cuda.set_device(rank)
        
        self.logger.info(f"Initialized distributed training: Rank {rank}/{world_size}")
    
    def create_model(self, trial: Optional[optuna.Trial] = None) -> nn.Module:
        """Create model based on configuration or hyperparameter trial"""
        if trial:
            # Hyperparameter optimization mode
            model_config = self._sample_model_hyperparameters(trial)
        else:
            # Standard configuration mode
            model_config = {
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'dropout': self.config.dropout,
                'embedding_dim': self.config.embedding_dim
            }
        
        # Select model architecture
        if self.config.model_type == "transformer":
            model = TransformerItineraryModel(**model_config)
        elif self.config.model_type == "graph":
            model = GraphNeuralPOINetwork(**model_config)
        elif self.config.model_type == "hierarchical":
            model = HierarchicalPolicyNetwork(**model_config)
        elif self.config.model_type == "uncertainty":
            model = UncertaintyAwareQNetwork(**model_config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Move to appropriate device
        device = torch.device(f"cuda:{self.rank}" if self.distributed else self.config.device)
        model = model.to(device)
        
        # Wrap with DistributedDataParallel if using multiple GPUs
        if self.distributed:
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        
        return model
    
    def create_algorithm(self, model: nn.Module, trial: Optional[optuna.Trial] = None) -> Any:
        """Create training algorithm based on configuration"""
        if trial:
            # Hyperparameter optimization mode
            algo_config = self._sample_algorithm_hyperparameters(trial)
        else:
            # Standard configuration mode
            algo_config = {
                'learning_rate': self.config.learning_rate,
                'gamma': self.config.gamma,
                'tau': self.config.tau,
                'batch_size': self.config.batch_size
            }
        
        # Select training algorithm
        if self.config.algorithm == "maddpg":
            algorithm = MADDPG(
                model=model,
                num_agents=self.config.num_agents,
                **algo_config
            )
        elif self.config.algorithm == "ppo":
            algorithm = PPO(
                model=model,
                **algo_config
            )
        elif self.config.algorithm == "hierarchical":
            algorithm = HierarchicalRL(
                model=model,
                **algo_config
            )
        elif self.config.algorithm == "constrained":
            algorithm = ConstrainedOptimization(
                model=model,
                capacity_constraints=self._get_capacity_constraints(),
                **algo_config
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        return algorithm
    
    def create_environment(self) -> MultiAgentTourismEnv:
        """Create multi-agent tourism environment"""
        env = MultiAgentTourismEnv(
            num_agents=self.config.num_agents,
            poi_data=self.train_data,
            max_steps=self.config.max_episode_steps,
            capacity_factor=self.config.poi_capacity_factor,
            crowding_penalty=self.config.crowding_penalty_weight,
            sustainability_weight=self.config.sustainability_weight
        )
        return env
    
    def train_epoch(
        self,
        model: nn.Module,
        algorithm: Any,
        env: MultiAgentTourismEnv,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch"""
        model.train()
        epoch_metrics = defaultdict(list)
        
        # Use mixed precision training if configured
        scaler = GradScaler() if self.config.mixed_precision else None
        
        progress_bar = tqdm(
            range(self.config.num_episodes // self.world_size),
            desc=f"Epoch {epoch} - Rank {self.rank}"
        ) if self.rank == 0 else range(self.config.num_episodes // self.world_size)
        
        for episode in progress_bar:
            # Reset environment
            states = env.reset()
            episode_rewards = []
            episode_losses = []
            
            for step in range(self.config.max_episode_steps):
                # Get actions from algorithm
                actions = algorithm.select_actions(states)
                
                # Environment step
                next_states, rewards, dones, infos = env.step(actions)
                
                # Store transition
                algorithm.store_transition(states, actions, rewards, next_states, dones)
                
                # Training step
                if len(algorithm.memory) >= self.config.batch_size:
                    if self.config.mixed_precision and scaler:
                        with autocast():
                            loss = algorithm.train_step()
                        scaler.scale(loss).backward()
                        scaler.step(algorithm.optimizer)
                        scaler.update()
                    else:
                        loss = algorithm.train_step()
                        loss.backward()
                        algorithm.optimizer.step()
                    
                    episode_losses.append(loss.item())
                
                episode_rewards.append(rewards.mean().item())
                states = next_states
                
                if dones.all():
                    break
            
            # Record episode metrics
            epoch_metrics['reward'].append(np.sum(episode_rewards))
            if episode_losses:
                epoch_metrics['loss'].append(np.mean(episode_losses))
            epoch_metrics['episode_length'].append(step + 1)
            
            # Log metrics periodically
            if episode % self.config.log_interval == 0 and self.rank == 0:
                self._log_training_metrics(epoch, episode, epoch_metrics)
        
        # Aggregate metrics across all processes
        if self.distributed:
            epoch_metrics = self._aggregate_metrics(epoch_metrics)
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def evaluate(
        self,
        model: nn.Module,
        algorithm: Any,
        data: pd.DataFrame,
        phase: str = "val"
    ) -> Dict[str, float]:
        """Evaluate model performance on validation or test data"""
        model.eval()
        eval_env = MultiAgentTourismEnv(
            num_agents=self.config.num_agents,
            poi_data=data,
            max_steps=self.config.max_episode_steps,
            capacity_factor=self.config.poi_capacity_factor,
            crowding_penalty=self.config.crowding_penalty_weight,
            sustainability_weight=self.config.sustainability_weight
        )
        
        eval_metrics = defaultdict(list)
        
        with torch.no_grad():
            for episode in range(100):  # Fixed number of evaluation episodes
                states = eval_env.reset()
                episode_rewards = []
                episode_sustainability = []
                episode_satisfaction = []
                
                for step in range(self.config.max_episode_steps):
                    actions = algorithm.select_actions(states, evaluate=True)
                    next_states, rewards, dones, infos = eval_env.step(actions)
                    
                    episode_rewards.append(rewards.mean().item())
                    episode_sustainability.append(infos.get('sustainability_score', 0))
                    episode_satisfaction.append(infos.get('satisfaction_score', 0))
                    
                    states = next_states
                    
                    if dones.all():
                        break
                
                eval_metrics['reward'].append(np.sum(episode_rewards))
                eval_metrics['sustainability'].append(np.mean(episode_sustainability))
                eval_metrics['satisfaction'].append(np.mean(episode_satisfaction))
                eval_metrics['episode_length'].append(step + 1)
        
        # Calculate advanced metrics
        advanced_metrics = self.metrics.calculate_advanced_metrics(
            eval_env.get_trajectories(),
            eval_env.get_poi_visits()
        )
        eval_metrics.update(advanced_metrics)
        
        return {k: np.mean(v) if isinstance(v, list) else v for k, v in eval_metrics.items()}
    
    def hyperparameter_optimization(self) -> Dict[str, Any]:
        """Perform hyperparameter optimization using Optuna"""
        self.logger.info("Starting hyperparameter optimization...")
        
        def objective(trial: optuna.Trial) -> float:
            # Create model and algorithm with sampled hyperparameters
            model = self.create_model(trial)
            algorithm = self.create_algorithm(model, trial)
            env = self.create_environment()
            
            # Train for reduced number of epochs
            for epoch in range(10):  # Reduced epochs for hyperopt
                train_metrics = self.train_epoch(model, algorithm, env, epoch)
                
                # Evaluate on validation set
                val_metrics = self.evaluate(model, algorithm, self.val_data, "val")
                
                # Report intermediate value for pruning
                trial.report(val_metrics['reward'], epoch)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return val_metrics['reward']
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.seed),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.hyperopt_trials,
            timeout=self.config.hyperopt_timeout,
            n_jobs=1  # Use 1 job per GPU
        )
        
        # Log best parameters
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        if self.config.use_wandb:
            wandb.log({"best_hyperparameters": best_params})
        
        if self.config.use_mlflow:
            mlflow.log_params(best_params)
        
        return best_params
    
    def _sample_model_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample model hyperparameters for Optuna trial"""
        return {
            'hidden_dim': trial.suggest_int('hidden_dim', 128, 512, step=64),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
            'embedding_dim': trial.suggest_int('embedding_dim', 64, 256, step=32)
        }
    
    def _sample_algorithm_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample algorithm hyperparameters for Optuna trial"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999, step=0.001),
            'tau': trial.suggest_float('tau', 0.001, 0.01, step=0.001),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        }
    
    def _get_capacity_constraints(self) -> Dict[int, int]:
        """Get POI capacity constraints from data"""
        poi_metadata = self.data_loader.load_poi_metadata()
        return {poi_id: capacity for poi_id, capacity in 
                zip(poi_metadata['poi_id'], poi_metadata['capacity'])}
    
    def _log_training_metrics(
        self,
        epoch: int,
        episode: int,
        metrics: Dict[str, List[float]]
    ) -> None:
        """Log training metrics to all tracking systems"""
        avg_metrics = {k: np.mean(v[-100:]) if v else 0 for k, v in metrics.items()}
        
        if self.config.use_wandb:
            wandb.log({
                f"train/{k}": v for k, v in avg_metrics.items()
            }, step=epoch * self.config.num_episodes + episode)
        
        if self.config.use_mlflow:
            for k, v in avg_metrics.items():
                mlflow.log_metric(f"train_{k}", v, step=epoch * self.config.num_episodes + episode)
        
        if self.config.use_tensorboard:
            for k, v in avg_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch * self.config.num_episodes + episode)
    
    def _aggregate_metrics(self, metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Aggregate metrics across distributed processes"""
        aggregated = {}
        for key, values in metrics.items():
            tensor = torch.tensor(values, device=f"cuda:{self.rank}")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            aggregated[key] = (tensor / self.world_size).cpu().tolist()
        return aggregated
    
    def save_model(self, model: nn.Module, algorithm: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{self.config.experiment_name}_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': algorithm.optimizer.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save to MLflow if configured
        if self.config.use_mlflow:
            mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")
    
    def run(self) -> None:
        """Main training loop"""
        self.logger.info("Starting main training loop...")
        
        # Set random seeds for reproducibility
        set_random_seeds(self.config.seed)
        
        # Hyperparameter optimization if configured
        best_hyperparams = None
        if self.config.use_hyperopt:
            best_hyperparams = self.hyperparameter_optimization()
        
        # Create model and algorithm with best hyperparameters or default config
        model = self.create_model()
        algorithm = self.create_algorithm(model)
        env = self.create_environment()
        
        # Load checkpoint if resuming
        start_epoch = 0
        if self.config.resume_from_checkpoint:
            checkpoint = load_checkpoint(self.config.resume_from_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            algorithm.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resumed from epoch {start_epoch}")
        
        # Main training loop
        for epoch in range(start_epoch, self.config.num_episodes):
            # Training
            train_metrics = self.train_epoch(model, algorithm, env, epoch)
            self.logger.info(f"Epoch {epoch} - Train metrics: {train_metrics}")
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(model, algorithm, self.val_data, "val")
                self.logger.info(f"Epoch {epoch} - Validation metrics: {val_metrics}")
                
                # Early stopping check
                if val_metrics['reward'] > self.best_val_performance:
                    self.best_val_performance = val_metrics['reward']
                    self.patience_counter = 0
                    self.save_model(model, algorithm, epoch, val_metrics)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
            
            # Periodic checkpoint saving
            if epoch % self.config.save_interval == 0:
                self.save_model(model, algorithm, epoch, train_metrics)
        
        # Final evaluation on test set
        self.logger.info("Running final evaluation on test set...")
        test_metrics = self.evaluate(model, algorithm, self.test_data, "test")
        self.logger.info(f"Test metrics: {test_metrics}")
        
        # Generate final visualizations
        self.visualizer.generate_final_report(test_metrics, env.get_trajectories())
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        if self.config.use_wandb:
            wandb.finish()
        
        if self.config.use_mlflow:
            mlflow.end_run()
        
        if self.config.use_tensorboard:
            self.writer.close()
        
        if self.distributed:
            dist.destroy_process_group()
        
        self.logger.info("Experiment completed successfully")


def distributed_worker(rank: int, world_size: int, config: ExperimentConfig) -> None:
    """Worker function for distributed training"""
    orchestrator = ExperimentOrchestrator(config)
    orchestrator.setup_distributed(rank, world_size)
    orchestrator.run()


def main():
    """Main entry point for the experiment"""
    parser = argparse.ArgumentParser(description="Multi-Agent RL for Sustainable Tourism")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, help='Override experiment name')
    parser.add_argument('--num_gpus', type=int, help='Override number of GPUs')
    parser.add_argument('--use_hyperopt', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override configuration with command-line arguments
    if args.experiment_name:
        config_dict['experiment_name'] = args.experiment_name
    if args.num_gpus:
        config_dict['num_gpus'] = args.num_gpus
    if args.use_hyperopt:
        config_dict['use_hyperopt'] = True
    if args.resume:
        config_dict['resume_from_checkpoint'] = args.resume
    
    # Create configuration object
    config = ExperimentConfig(**config_dict)
    
    # Run experiment
    if config.num_gpus > 1:
        # Distributed training
        mp.spawn(
            distributed_worker,
            args=(config.num_gpus, config),
            nprocs=config.num_gpus,
            join=True
        )
    else:
        # Single GPU or CPU training
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()


if __name__ == "__main__":
    main()