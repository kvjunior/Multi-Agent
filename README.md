# Multi-Agent Reinforcement Learning for Sustainable Tourism Management

## Abstract

This repository contains the implementation of a multi-agent reinforcement learning framework for sustainable tourism management, addressing the critical challenge of overtourism through distributed coordination mechanisms. The system achieves a 34.5% improvement in system-wide efficiency while reducing distributional inequality (Gini coefficient) from 0.478 to 0.236, demonstrating that algorithmic coordination can balance individual tourist satisfaction with destination sustainability objectives.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Experimental Reproduction](#experimental-reproduction)
- [API Documentation](#api-documentation)
- [Performance Benchmarks](#performance-benchmarks)

## System Requirements

### Hardware Requirements

The framework requires substantial computational resources for training multi-agent systems at scale. The following specifications represent minimum and recommended configurations:

**Minimum Configuration:**
- CPU: Intel Xeon E5-2690 v4 or equivalent (14+ cores)
- GPU: NVIDIA V100 16GB (single GPU)
- RAM: 64GB DDR4
- Storage: 500GB SSD for datasets and checkpoints
- Network: 1 Gbps for distributed training

**Recommended Configuration:**
- CPU: AMD EPYC 7742 or equivalent (32+ cores)
- GPU: 4x NVIDIA A100 40GB with NVLink
- RAM: 256GB DDR4
- Storage: 2TB NVMe SSD
- Network: 10 Gbps for distributed training

### Software Dependencies

The framework requires Python 3.8 or higher with CUDA 11.3+ for GPU acceleration. All dependencies are specified in `requirements.txt` with exact version numbers to ensure reproducibility:

```
torch==1.13.0+cu117
torch-geometric==2.2.0
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.0
scipy==1.10.0
networkx==3.0
matplotlib==3.6.3
seaborn==0.12.2
tqdm==4.64.1
wandb==0.13.9
mlflow==2.1.1
optuna==3.1.0
pyyaml==6.0
cvxpy==1.3.0
```

## Installation

### Standard Installation

Clone the repository and install dependencies through the following procedure:

```bash
git clone https://github.com/yourinstitution/tourism-marl.git
cd tourism-marl
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Docker Installation

For containerized deployment ensuring complete reproducibility, utilize the provided Docker configuration:

```bash
docker build -t tourism-marl:latest .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results tourism-marl:latest
```

### Distributed Installation

For multi-node distributed training, configure the cluster environment:

```bash
# On each node
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=12355
export WORLD_SIZE=<total_number_of_nodes>
export RANK=<node_rank>

# Install MPI backend
conda install -c conda-forge mpi4py
pip install -r requirements-distributed.txt
```

## Dataset Preparation

### Data Structure

The framework expects data organized in the following hierarchical structure:

```
data/
├── raw/
│   ├── poi_metadata.csv          # POI characteristics and capacities
│   ├── historical_visits.csv     # Tourist trajectory records
│   ├── crowding_patterns.csv     # Temporal occupancy data
│   └── contextual_factors.csv    # Weather and event data
├── processed/
│   ├── train/                    # 70% of chronological data
│   ├── val/                      # 15% of chronological data
│   └── test/                     # 15% of chronological data
└── cache/                         # Preprocessed features
```

### Data Processing Pipeline

Execute the preprocessing pipeline to generate training-ready datasets:

```python
from utilities import DataLoader

# Initialize data loader with caching
data_loader = DataLoader(data_path="./data", cache_dir="./data/cache")

# Load and validate POI metadata
poi_metadata = data_loader.load_poi_metadata(use_cache=True)
print(f"Loaded {len(poi_metadata)} POIs with capacity constraints")

# Process historical visits with chunking for large datasets
historical_visits = data_loader.load_historical_visits(chunk_size=100000)
print(f"Processed {len(historical_visits):,} visit records")

# Generate augmented dataset for improved coverage
augmented_data = data_loader.augment_trajectories(
    historical_visits,
    perturbation_factor=0.2,
    temporal_shift_hours=2
)
```

### Verona Case Study Data

The repository includes preprocessed data from Verona, Italy, encompassing 18 cultural attractions:

```python
# Load Verona-specific configuration
from environments import create_verona_environment

env_config = {
    'num_agents': 10,
    'max_steps': 100,
    'capacity_factor': 1.0,
    'crowding_penalty': 0.3,
    'sustainability_weight': 0.2
}

env = create_verona_environment(env_config)
```

## Configuration

### Experiment Configuration

Modify `config.yaml` to customize experimental parameters. Critical parameters affecting performance include:

```yaml
algorithm:
  type: "maddpg"  # Options: maddpg, ppo, hierarchical, constrained
  learning_rate: 1e-4
  batch_size: 256
  gamma: 0.99
  tau: 0.001
  gradient_clip: 10.0
  
environment:
  num_agents: 10
  max_episode_steps: 100
  capacity_factor: 1.0
  crowding_penalty_weight: 0.3
  sustainability_weight: 0.2
  
model:
  type: "transformer"  # Options: transformer, graph, hierarchical, uncertainty
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1
```

### Hyperparameter Optimization

Execute automated hyperparameter search using Optuna:

```python
from main import ExperimentOrchestrator, ExperimentConfig

config = ExperimentConfig(
    experiment_name="hyperopt_search",
    use_hyperopt=True,
    hyperopt_trials=100,
    hyperopt_timeout=3600
)

orchestrator = ExperimentOrchestrator(config)
best_params = orchestrator.hyperparameter_optimization()
print(f"Optimal parameters: {best_params}")
```

## Training

### Single-GPU Training

Initiate training with default configuration:

```bash
python main.py --config config.yaml --experiment_name baseline_maddpg
```

### Distributed Training

Launch distributed training across multiple GPUs:

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    main.py --config config.yaml --num_gpus 4 --distributed
```

### Monitoring Training Progress

Training metrics are logged to multiple platforms simultaneously:

```python
# TensorBoard visualization
tensorboard --logdir experiments/logs/tensorboard

# Weights & Biases tracking
wandb login
python main.py --config config.yaml --use_wandb

# MLflow tracking
mlflow ui --host 0.0.0.0 --port 5000
```

## Evaluation

### Performance Evaluation

Evaluate trained models on test data:

```python
from algorithms import create_algorithm
from models import create_model
from utilities import MetricsCalculator, load_checkpoint

# Load trained model
checkpoint = load_checkpoint("experiments/checkpoints/best_model.pt")
model = create_model("transformer", checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize evaluation
algorithm = create_algorithm("maddpg", model, checkpoint['config'])
metrics_calculator = MetricsCalculator()

# Compute comprehensive metrics
results = metrics_calculator.calculate_advanced_metrics(
    trajectories=test_trajectories,
    poi_visits=poi_visit_counts
)

print(f"Gini Coefficient: {results['gini_coefficient']:.3f}")
print(f"Average Satisfaction: {results['avg_satisfaction']:.2f}")
print(f"Time Utilization: {results['time_utilization']:.1%}")
```

### Robustness Testing

Assess model robustness under perturbations:

```bash
python evaluate_robustness.py --checkpoint best_model.pt \
    --noise_levels 0.0,0.1,0.2,0.3 \
    --missing_data_rates 0.0,0.15,0.3 \
    --adversarial_scenarios flash_crowd,attraction_closure
```

## Deployment

### API Server Deployment

Deploy the trained model as a RESTful API service:

```python
from flask import Flask, request, jsonify
from deployment import InferenceServer

app = Flask(__name__)
server = InferenceServer(checkpoint_path="best_model.pt")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    observation = data['observation']
    agent_id = data['agent_id']
    
    action = server.get_recommendation(observation, agent_id)
    return jsonify({'recommended_poi': action, 'confidence': 0.95})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
```

### Cloud Deployment

Deploy to AWS using the provided CloudFormation template:

```bash
aws cloudformation create-stack \
    --stack-name tourism-marl \
    --template-body file://deployment/cloudformation.yaml \
    --parameters ParameterKey=InstanceType,ParameterValue=p3.8xlarge \
                 ParameterKey=MinInstances,ParameterValue=2 \
                 ParameterKey=MaxInstances,ParameterValue=20
```

## Experimental Reproduction

To reproduce the paper's main results, execute the following experimental pipeline:

```bash
# Experiment 1: Algorithm Comparison
python run_experiments.py --experiment algorithm_comparison \
    --algorithms maddpg,ppo,hierarchical,constrained \
    --num_runs 100 --seed 42

# Experiment 2: Ablation Study
python run_experiments.py --experiment ablation \
    --base_model maddpg \
    --ablate_components graph,transformer,multi_agent,uncertainty \
    --num_runs 50

# Experiment 3: Scalability Analysis
python run_experiments.py --experiment scalability \
    --num_agents 10,50,100,500,1000,5000,10000 \
    --measure_memory --measure_latency

# Experiment 4: Generalization Testing
python run_experiments.py --experiment generalization \
    --test_contexts seasonal_shift,cultural_variation,venue_type \
    --cross_validation 5-fold
```

Expected computational requirements for complete reproduction:
- Training time: 14.2 hours (MADDPG, 4x A100 GPUs)
- Total experiments: approximately 200 GPU-hours
- Storage: 150GB for checkpoints and logs

## API Documentation

### Core Classes

```python
class MultiAgentTourismEnv:
    """
    Multi-agent environment for tourism simulation.
    
    Args:
        num_agents (int): Number of concurrent tourist agents
        poi_data (DataFrame): POI metadata including capacities
        max_steps (int): Maximum episode length
        capacity_factor (float): Scaling factor for POI capacities
        crowding_penalty (float): Weight for crowding penalties
        sustainability_weight (float): Weight for sustainability objectives
    
    Methods:
        reset(): Initialize environment state
        step(actions): Execute joint action and return observations
        get_metrics(): Retrieve current performance metrics
    """
    
class MADDPG(BaseAlgorithm):
    """
    Multi-Agent Deep Deterministic Policy Gradient implementation.
    
    Args:
        model (nn.Module): Actor-critic network architecture
        config (AlgorithmConfig): Training configuration
    
    Methods:
        select_actions(states, evaluate=False): Generate actions
        train_step(): Execute single training iteration
        store_transition(*args): Add experience to replay buffer
    """
```

### Utility Functions

```python
def create_model(model_type: str, config: Dict) -> nn.Module:
    """
    Factory function for model instantiation.
    
    Args:
        model_type: One of ['transformer', 'graph', 'hierarchical', 'uncertainty']
        config: Model configuration dictionary
    
    Returns:
        Initialized neural network model
    """

def calculate_trajectory_similarity(traj1: List[int], traj2: List[int]) -> float:
    """
    Compute similarity between POI trajectories using edit distance.
    
    Args:
        traj1: First trajectory as POI ID sequence
        traj2: Second trajectory as POI ID sequence
    
    Returns:
        Similarity score in range [0, 1]
    """
```

## Performance Benchmarks

Performance metrics across standard evaluation scenarios:

| Metric | Single-Agent | MADDPG | PPO | Hierarchical | Constrained |
|--------|-------------|---------|-----|--------------|-------------|
| Average Reward | 0.615 (0.074) | **0.827 (0.052)** | 0.794 (0.061) | 0.812 (0.056) | 0.768 (0.068) |
| Gini Coefficient | 0.478 (0.042) | 0.236 (0.028) | 0.252 (0.031) | 0.218 (0.024) | **0.194 (0.021)** |
| Satisfaction (0-10) | 6.24 (0.89) | **8.43 (1.21)** | 8.05 (1.34) | 8.28 (1.18) | 7.82 (1.42) |
| Time Utilization | 57.5% | 71.3% | 70.0% | **72.5%** | 67.5% |
| Training Time (hours) | 8.7 | 14.2 | 11.8 | 16.5 | 10.3 |

Values represent mean (standard deviation) across 100 independent experimental runs.
