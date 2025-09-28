"""
models.py - Neural Network Architectures for Multi-Agent Sustainable Tourism
Academic implementation for Information Technology & Tourism journal submission
Authors: [Your Name]
Institution: [Your Institution]
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Normal, Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np


class ActivationType(Enum):
    """Enumeration of activation function types"""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"


def get_activation(activation_type: Union[str, ActivationType]) -> nn.Module:
    """Factory function for activation layers"""
    if isinstance(activation_type, str):
        activation_type = ActivationType(activation_type)
    
    activations = {
        ActivationType.RELU: nn.ReLU(),
        ActivationType.GELU: nn.GELU(),
        ActivationType.SWISH: nn.SiLU(),
        ActivationType.MISH: nn.Mish()
    }
    return activations[activation_type]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SpatioTemporalEmbedding(nn.Module):
    """Combined spatial and temporal embedding for POI sequences"""
    
    def __init__(
        self,
        num_pois: int,
        embedding_dim: int,
        max_time_slots: int = 48,
        max_days: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # POI embedding
        self.poi_embedding = nn.Embedding(num_pois, embedding_dim)
        
        # Temporal embeddings
        self.time_slot_embedding = nn.Embedding(max_time_slots, embedding_dim // 4)
        self.day_embedding = nn.Embedding(max_days, embedding_dim // 4)
        
        # Spatial encoding (learned from coordinates)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # Projection layer
        self.projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        poi_ids: torch.Tensor,
        coordinates: torch.Tensor,
        time_slots: torch.Tensor,
        days: torch.Tensor
    ) -> torch.Tensor:
        """Generate spatio-temporal embeddings"""
        # Get individual embeddings
        poi_emb = self.poi_embedding(poi_ids)
        time_emb = self.time_slot_embedding(time_slots)
        day_emb = self.day_embedding(days)
        spatial_emb = self.spatial_encoder(coordinates)
        
        # Concatenate all embeddings
        combined = torch.cat([poi_emb, spatial_emb, time_emb, day_emb], dim=-1)
        
        # Project to final dimension
        output = self.projection(combined)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class TransformerItineraryModel(nn.Module):
    """Transformer-based model for sequential itinerary recommendation
    
    This model uses multi-head self-attention to capture complex dependencies
    between POI visits in a sequence, considering both spatial and temporal contexts.
    """
    
    def __init__(
        self,
        num_pois: int = 18,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        embedding_dim: int = 128,
        max_sequence_length: int = 20,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.num_pois = num_pois
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Spatio-temporal embedding layer
        self.embedding = SpatioTemporalEmbedding(
            num_pois=num_pois,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=embedding_dim,
            max_len=max_sequence_length,
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        
        # Context integration layers
        self.context_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # Weather, time, crowding features
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Output layers for different tasks
        self.poi_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pois)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sustainability_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        poi_sequence: torch.Tensor,
        coordinates: torch.Tensor,
        time_slots: torch.Tensor,
        days: torch.Tensor,
        context_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer model
        
        Args:
            poi_sequence: Tensor of shape (batch_size, seq_len) containing POI IDs
            coordinates: Tensor of shape (batch_size, seq_len, 2) with lat/lon
            time_slots: Tensor of shape (batch_size, seq_len) with time slot indices
            days: Tensor of shape (batch_size, seq_len) with day indices
            context_features: Tensor of shape (batch_size, context_dim) with context
            mask: Optional attention mask for padding
        
        Returns:
            Dictionary containing predictions for next POI, value, and sustainability
        """
        batch_size, seq_len = poi_sequence.shape
        
        # Generate embeddings
        embeddings = self.embedding(poi_sequence, coordinates, time_slots, days)
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Encode context and add to embeddings
        context_encoded = self.context_encoder(context_features)
        context_encoded = context_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        embeddings = embeddings + context_encoded
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        
        # Pool sequence representation (use last non-padded position)
        if mask is not None:
            lengths = (~mask).sum(dim=1)
            sequence_repr = torch.stack([
                transformer_output[i, lengths[i] - 1] 
                for i in range(batch_size)
            ])
        else:
            sequence_repr = transformer_output[:, -1, :]
        
        # Generate predictions
        next_poi_logits = self.poi_predictor(sequence_repr)
        value_estimate = self.value_head(sequence_repr)
        sustainability_score = self.sustainability_head(sequence_repr)
        
        return {
            'next_poi_logits': next_poi_logits,
            'value': value_estimate,
            'sustainability': sustainability_score,
            'sequence_embeddings': transformer_output
        }


class GraphNeuralPOINetwork(nn.Module):
    """Graph Neural Network for modeling POI relationships and spatial dynamics
    
    This model treats POIs as nodes in a graph where edges represent spatial
    proximity, travel patterns, and functional relationships.
    """
    
    def __init__(
        self,
        num_pois: int = 18,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        embedding_dim: int = 128,
        edge_dim: int = 32,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.num_pois = num_pois
        self.hidden_dim = hidden_dim
        
        # Node feature encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(10, embedding_dim),  # POI features
            get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # Edge feature encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, edge_dim),  # Distance, time, correlation features
            get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = embedding_dim if i == 0 else hidden_dim
            out_channels = hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
        
        # Message passing for crowd dynamics
        self.crowd_propagation = CrowdDynamicsLayer(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Readout layers
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pois)
        )
        
        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Linear(hidden_dim // 2, num_pois),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        current_state: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through graph neural network
        
        Args:
            node_features: Node feature matrix (num_nodes, feature_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_features: Edge feature matrix (num_edges, edge_feature_dim)
            current_state: Current agent states (batch_size, state_dim)
            batch: Batch assignment for nodes
        
        Returns:
            Dictionary with action logits and auxiliary predictions
        """
        # Encode node and edge features
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        
        # Apply graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            x_res = x
            x = gat_layer(x, edge_index)
            x = F.relu(x)
            
            # Residual connection for deeper layers
            if i > 0 and x.shape == x_res.shape:
                x = x + x_res
        
        # Apply crowd dynamics propagation
        x = self.crowd_propagation(x, edge_index, edge_attr)
        
        # Global graph representation
        if batch is not None:
            graph_repr = global_mean_pool(x, batch)
        else:
            graph_repr = x.mean(dim=0, keepdim=True)
        
        graph_repr = self.graph_pool(graph_repr)
        
        # Combine with current state
        combined = torch.cat([graph_repr.expand(current_state.size(0), -1), current_state], dim=-1)
        
        # Generate predictions
        action_logits = self.action_head(combined)
        capacity_usage = self.capacity_predictor(x)
        
        return {
            'action_logits': action_logits,
            'capacity_usage': capacity_usage,
            'node_embeddings': x,
            'graph_embedding': graph_repr
        }


class CrowdDynamicsLayer(MessagePassing):
    """Custom message passing layer for modeling crowd dynamics"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__(aggr='mean')
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Propagate crowd information through graph"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute messages between nodes"""
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)
    
    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Update node representations"""
        update_input = torch.cat([x, aggr_out], dim=-1)
        updated = self.update_mlp(update_input)
        return self.layer_norm(updated + x)


class HierarchicalPolicyNetwork(nn.Module):
    """Hierarchical reinforcement learning model for multi-day itinerary planning
    
    Implements a two-level hierarchy:
    - High-level: Decides daily themes/zones
    - Low-level: Generates specific POI sequences within constraints
    """
    
    def __init__(
        self,
        num_pois: int = 18,
        num_zones: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        embedding_dim: int = 128,
        max_days: int = 7,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.num_pois = num_pois
        self.num_zones = num_zones
        self.max_days = max_days
        
        # High-level policy network (Manager)
        self.high_level_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.zone_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_zones)
        )
        
        self.theme_encoder = nn.Sequential(
            nn.Linear(num_zones + 10, embedding_dim),  # Zone + context features
            get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # Low-level policy network (Worker)
        self.low_level_encoder = nn.GRU(
            input_size=embedding_dim + num_zones,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.poi_selector = nn.Sequential(
            nn.Linear(hidden_dim + num_zones, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pois)
        )
        
        # Subgoal critic for high-level policy
        self.subgoal_critic = nn.Sequential(
            nn.Linear(hidden_dim + num_zones, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Intrinsic reward predictor for low-level policy
        self.intrinsic_reward = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Multi-day planning components
        self.day_encoder = nn.Embedding(max_days, embedding_dim // 2)
        self.variety_scorer = nn.Sequential(
            nn.Linear(hidden_dim * max_days, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU)):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                else:
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)
    
    def forward_high_level(
        self,
        global_state: torch.Tensor,
        day_index: torch.Tensor,
        historical_zones: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """High-level policy forward pass
        
        Args:
            global_state: Overall trip context (batch_size, state_dim)
            day_index: Current day index (batch_size,)
            historical_zones: Previously selected zones (batch_size, num_prev_days)
        
        Returns:
            Zone selection and subgoal value
        """
        batch_size = global_state.size(0)
        
        # Encode day information
        day_embedding = self.day_encoder(day_index)
        
        # Combine with global state
        high_level_input = torch.cat([global_state, day_embedding], dim=-1)
        high_level_input = high_level_input.unsqueeze(1)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.high_level_encoder(high_level_input)
        high_level_repr = h_n[-1]
        
        # Select zone/theme for the day
        zone_logits = self.zone_selector(high_level_repr)
        zone_probs = F.softmax(zone_logits, dim=-1)
        
        # Evaluate subgoal value
        zone_one_hot = F.one_hot(zone_logits.argmax(dim=-1), num_classes=self.num_zones).float()
        critic_input = torch.cat([high_level_repr, zone_one_hot], dim=-1)
        subgoal_value = self.subgoal_critic(critic_input)
        
        # Calculate variety score if historical zones provided
        variety_score = None
        if historical_zones is not None:
            all_days_repr = torch.cat([high_level_repr, historical_zones], dim=-1)
            variety_score = self.variety_scorer(all_days_repr)
        
        return {
            'zone_logits': zone_logits,
            'zone_probs': zone_probs,
            'subgoal_value': subgoal_value,
            'variety_score': variety_score,
            'high_level_repr': high_level_repr
        }
    
    def forward_low_level(
        self,
        local_state: torch.Tensor,
        zone_assignment: torch.Tensor,
        poi_sequence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Low-level policy forward pass
        
        Args:
            local_state: Current state within day (batch_size, seq_len, state_dim)
            zone_assignment: Assigned zone from high-level (batch_size, num_zones)
            poi_sequence: Optional previous POI sequence
        
        Returns:
            POI selection logits and intrinsic rewards
        """
        batch_size, seq_len = local_state.shape[:2]
        
        # Expand zone assignment for sequence
        zone_expanded = zone_assignment.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with local state
        low_level_input = torch.cat([local_state, zone_expanded], dim=-1)
        
        # Process through GRU
        gru_out, h_n = self.low_level_encoder(low_level_input)
        
        # Generate POI selections
        poi_input = torch.cat([gru_out, zone_expanded], dim=-1)
        poi_logits = self.poi_selector(poi_input)
        
        # Calculate intrinsic rewards
        intrinsic_input = torch.cat([gru_out, local_state], dim=-1)
        intrinsic_rewards = self.intrinsic_reward(intrinsic_input)
        
        return {
            'poi_logits': poi_logits,
            'intrinsic_rewards': intrinsic_rewards,
            'low_level_repr': gru_out
        }
    
    def forward(
        self,
        global_state: torch.Tensor,
        local_state: torch.Tensor,
        day_index: torch.Tensor,
        historical_zones: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Complete hierarchical forward pass"""
        
        # High-level decision
        high_level_output = self.forward_high_level(global_state, day_index, historical_zones)
        
        # Get zone assignment
        zone_assignment = F.one_hot(
            high_level_output['zone_logits'].argmax(dim=-1),
            num_classes=self.num_zones
        ).float()
        
        # Low-level decision
        low_level_output = self.forward_low_level(local_state, zone_assignment)
        
        # Combine outputs
        return {
            **high_level_output,
            **low_level_output,
            'zone_assignment': zone_assignment
        }


class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty quantification"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Prior distributions
        self.weight_prior = Normal(0, prior_std)
        self.bias_prior = Normal(0, prior_std)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_normal_(self.weight_mu, nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, -3)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional sampling"""
        
        if sample:
            # Sample weights and biases
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.randn_like(self.bias_mu)
            
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)
            
            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon
            
            # Calculate KL divergence
            weight_kl = self._kl_divergence(weight, self.weight_mu, weight_sigma)
            bias_kl = self._kl_divergence(bias, self.bias_mu, bias_sigma)
            kl = weight_kl + bias_kl
        else:
            # Use mean parameters
            weight = self.weight_mu
            bias = self.bias_mu
            kl = torch.tensor(0.0)
        
        output = F.linear(x, weight, bias)
        return output, kl
    
    def _kl_divergence(
        self,
        sample: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """Calculate KL divergence between posterior and prior"""
        posterior = Normal(mu, sigma)
        kl = torch.distributions.kl_divergence(posterior, self.weight_prior).sum()
        return kl


class UncertaintyAwareQNetwork(nn.Module):
    """Q-Network with Bayesian layers for uncertainty quantification
    
    This model provides uncertainty estimates for action values, enabling
    risk-aware decision making in tourism recommendations.
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 18,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        prior_std: float = 1.0,
        num_ensemble: int = 5,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        
        # Ensemble of Bayesian networks
        self.ensemble = nn.ModuleList()
        
        for _ in range(num_ensemble):
            layers = []
            
            # Input layer
            layers.append(BayesianLinear(state_dim, hidden_dim, prior_std))
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(BayesianLinear(hidden_dim, hidden_dim, prior_std))
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(dropout))
            
            # Output layer
            layers.append(BayesianLinear(hidden_dim, action_dim, prior_std))
            
            self.ensemble.append(nn.ModuleList(layers))
        
        # Uncertainty aggregation network
        self.uncertainty_head = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )
        
        # Risk-sensitive value head
        self.risk_head = nn.Sequential(
            nn.Linear(action_dim * 3, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty quantification
        
        Args:
            state: Input state tensor (batch_size, state_dim)
            num_samples: Number of Monte Carlo samples for uncertainty
        
        Returns:
            Dictionary containing Q-values, uncertainties, and risk measures
        """
        batch_size = state.size(0)
        
        # Collect predictions from ensemble
        ensemble_outputs = []
        total_kl = 0
        
        for network in self.ensemble:
            network_samples = []
            
            for _ in range(num_samples):
                x = state
                sample_kl = 0
                
                for i, layer in enumerate(network):
                    if isinstance(layer, BayesianLinear):
                        x, kl = layer(x, sample=True)
                        sample_kl += kl
                    else:
                        x = layer(x)
                
                network_samples.append(x)
                total_kl += sample_kl
            
            # Average over samples
            network_output = torch.stack(network_samples).mean(dim=0)
            ensemble_outputs.append(network_output)
        
        # Stack ensemble predictions
        ensemble_outputs = torch.stack(ensemble_outputs)  # (num_ensemble, batch_size, action_dim)
        
        # Calculate mean and variance
        q_values_mean = ensemble_outputs.mean(dim=0)
        q_values_var = ensemble_outputs.var(dim=0)
        
        # Epistemic uncertainty (variance across ensemble)
        epistemic_uncertainty = q_values_var
        
        # Aleatoric uncertainty (learned)
        uncertainty_input = torch.cat([q_values_mean, q_values_var], dim=-1)
        aleatoric_uncertainty = self.uncertainty_head(uncertainty_input)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Risk-adjusted values
        risk_input = torch.cat([q_values_mean, epistemic_uncertainty, aleatoric_uncertainty], dim=-1)
        risk_adjustment = self.risk_head(risk_input)
        
        # Lower confidence bound for exploration
        lcb = q_values_mean - 2.0 * torch.sqrt(total_uncertainty)
        
        # Upper confidence bound for optimistic exploration
        ucb = q_values_mean + 2.0 * torch.sqrt(total_uncertainty)
        
        return {
            'q_values': q_values_mean,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'lower_confidence_bound': lcb,
            'upper_confidence_bound': ucb,
            'risk_adjustment': risk_adjustment,
            'kl_divergence': total_kl / (self.num_ensemble * num_samples)
        }
    
    def get_action(
        self,
        state: torch.Tensor,
        exploration_strategy: str = "ucb"
    ) -> torch.Tensor:
        """Select action based on uncertainty-aware strategy
        
        Args:
            state: Input state
            exploration_strategy: One of "ucb", "lcb", "thompson"
        
        Returns:
            Selected action indices
        """
        output = self.forward(state)
        
        if exploration_strategy == "ucb":
            action_values = output['upper_confidence_bound']
        elif exploration_strategy == "lcb":
            action_values = output['lower_confidence_bound']
        elif exploration_strategy == "thompson":
            # Thompson sampling from posterior
            std = torch.sqrt(output['total_uncertainty'])
            action_values = Normal(output['q_values'], std).sample()
        else:
            action_values = output['q_values']
        
        return action_values.argmax(dim=-1)


# Utility function for model selection
def create_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """Factory function for model creation
    
    Args:
        model_type: One of "transformer", "graph", "hierarchical", "uncertainty"
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    """
    models = {
        'transformer': TransformerItineraryModel,
        'graph': GraphNeuralPOINetwork,
        'hierarchical': HierarchicalPolicyNetwork,
        'uncertainty': UncertaintyAwareQNetwork
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**config)