"""
utilities.py - Data Processing, Metrics, and Visualization for Sustainable Tourism
Academic implementation for Information Technology & Tourism journal submission
Authors: [Your Name]
Institution: [Your Institution]
"""

import numpy as np
import pandas as pd
import torch
import random
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import logging


# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class DataLoader:
    """Comprehensive data loading and preprocessing for tourism datasets.
    
    This class handles the efficient loading and preprocessing of large-scale tourism data,
    including POI metadata, historical visit records, crowding patterns, and contextual factors.
    The implementation uses chunked processing and vectorized operations to handle millions
    of records efficiently.
    """
    
    def __init__(self, data_path: str, cache_dir: Optional[str] = None):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path / '.cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        
    def load_poi_metadata(self, use_cache: bool = True) -> pd.DataFrame:
        """Load and preprocess POI metadata with caching support.
        
        This method loads POI information including coordinates, categories, capacities,
        and operational parameters. Data is cached for subsequent loads to improve performance.
        """
        cache_file = self.cache_dir / 'poi_metadata.pkl'
        
        if use_cache and cache_file.exists():
            self.logger.info("Loading POI metadata from cache")
            return pd.read_pickle(cache_file)
        
        poi_file = self.data_path / 'poi_metadata.csv'
        if not poi_file.exists():
            raise FileNotFoundError(f"POI metadata file not found: {poi_file}")
        
        poi_data = pd.read_csv(poi_file)
        
        # Validate and clean data
        required_columns = ['poi_id', 'name', 'latitude', 'longitude', 'capacity']
        missing_columns = set(required_columns) - set(poi_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        poi_data['category'] = poi_data.get('category', 'general').fillna('general')
        poi_data['visit_duration'] = poi_data.get('visit_duration', 30).fillna(30)
        poi_data['popularity'] = poi_data.get('popularity', 0.5).fillna(0.5)
        poi_data['sustainability'] = poi_data.get('sustainability', 0.5).fillna(0.5)
        
        # Add derived features
        poi_data['location_cluster'] = self._cluster_locations(
            poi_data[['latitude', 'longitude']].values
        )
        poi_data['capacity_category'] = pd.qcut(
            poi_data['capacity'], 
            q=4, 
            labels=['small', 'medium', 'large', 'xlarge']
        )
        
        # Cache processed data
        poi_data.to_pickle(cache_file)
        self.logger.info(f"Loaded {len(poi_data)} POIs")
        
        return poi_data
    
    def load_historical_visits(self, chunk_size: int = 100000) -> pd.DataFrame:
        """Load historical visit data with chunked processing for large datasets.
        
        This method efficiently processes millions of visit records using chunked reading
        and incremental processing to manage memory usage.
        """
        visits_file = self.data_path / 'historical_visits.csv'
        if not visits_file.exists():
            raise FileNotFoundError(f"Historical visits file not found: {visits_file}")
        
        processed_chunks = []
        total_records = 0
        
        # Process data in chunks for memory efficiency
        for chunk in pd.read_csv(visits_file, chunksize=chunk_size):
            # Clean and validate chunk
            chunk = self._preprocess_visit_chunk(chunk)
            processed_chunks.append(chunk)
            total_records += len(chunk)
            
            if total_records % 500000 == 0:
                self.logger.info(f"Processed {total_records:,} visit records")
        
        # Combine all chunks
        visits_data = pd.concat(processed_chunks, ignore_index=True)
        
        # Add derived temporal features
        visits_data = self._add_temporal_features(visits_data)
        
        # Create user profiles
        visits_data = self._create_user_profiles(visits_data)
        
        self.logger.info(f"Loaded {len(visits_data):,} total visit records")
        return visits_data
    
    def load_crowding_patterns(self) -> pd.DataFrame:
        """Load and process crowding pattern data.
        
        This method loads historical crowding patterns and generates predictive features
        for crowd forecasting.
        """
        crowding_file = self.data_path / 'crowding_patterns.csv'
        if not crowding_file.exists():
            raise FileNotFoundError(f"Crowding patterns file not found: {crowding_file}")
        
        crowding_data = pd.read_csv(crowding_file)
        
        # Parse datetime and add temporal features
        crowding_data['datetime'] = pd.to_datetime(crowding_data['datetime'])
        crowding_data['hour'] = crowding_data['datetime'].dt.hour
        crowding_data['day_of_week'] = crowding_data['datetime'].dt.dayofweek
        crowding_data['month'] = crowding_data['datetime'].dt.month
        
        # Calculate rolling statistics for trend analysis
        crowding_data = crowding_data.sort_values(['poi_id', 'datetime'])
        for poi_id in crowding_data['poi_id'].unique():
            poi_mask = crowding_data['poi_id'] == poi_id
            crowding_data.loc[poi_mask, 'crowding_ma_24h'] = (
                crowding_data.loc[poi_mask, 'crowding_level']
                .rolling(window=24, min_periods=1).mean()
            )
            crowding_data.loc[poi_mask, 'crowding_std_24h'] = (
                crowding_data.loc[poi_mask, 'crowding_level']
                .rolling(window=24, min_periods=1).std()
            )
        
        return crowding_data
    
    def load_contextual_factors(self) -> pd.DataFrame:
        """Load contextual factors including weather, events, and seasonal patterns."""
        context_file = self.data_path / 'contextual_factors.csv'
        if not context_file.exists():
            raise FileNotFoundError(f"Contextual factors file not found: {context_file}")
        
        context_data = pd.read_csv(context_file)
        context_data['date'] = pd.to_datetime(context_data['date'])
        
        # Add derived features
        context_data['is_weekend'] = context_data['date'].dt.dayofweek.isin([5, 6])
        context_data['season'] = context_data['date'].dt.month.map(self._get_season)
        
        # Encode weather conditions
        weather_encoding = {
            'sunny': 3,
            'cloudy': 2,
            'rainy': 1,
            'stormy': 0
        }
        context_data['weather_score'] = context_data['weather'].map(weather_encoding)
        
        return context_data
    
    def preprocess_data(self, poi_metadata: pd.DataFrame, historical_visits: pd.DataFrame,
                       crowding_patterns: pd.DataFrame, contextual_factors: pd.DataFrame) -> pd.DataFrame:
        """Integrate and preprocess all data sources into a unified dataset.
        
        This method performs comprehensive data integration, feature engineering,
        and normalization to prepare data for model training.
        """
        # Merge visit data with POI metadata
        merged_data = historical_visits.merge(
            poi_metadata[['poi_id', 'latitude', 'longitude', 'category', 'capacity']],
            on='poi_id',
            how='left'
        )
        
        # Add crowding information
        merged_data = self._add_crowding_features(merged_data, crowding_patterns)
        
        # Add contextual factors
        merged_data = self._add_contextual_features(merged_data, contextual_factors)
        
        # Create interaction features
        merged_data = self._create_interaction_features(merged_data)
        
        # Normalize numerical features
        numerical_columns = ['latitude', 'longitude', 'capacity', 'visit_duration',
                           'temperature', 'crowding_level']
        for col in numerical_columns:
            if col in merged_data.columns:
                merged_data[f'{col}_normalized'] = self.scaler.fit_transform(
                    merged_data[[col]]
                )
        
        return merged_data
    
    def _preprocess_visit_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a chunk of visit data."""
        # Remove duplicates
        chunk = chunk.drop_duplicates(subset=['user_id', 'poi_id', 'timestamp'])
        
        # Parse timestamps
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        
        # Sort by user and timestamp
        chunk = chunk.sort_values(['user_id', 'timestamp'])
        
        # Calculate visit duration
        chunk['next_timestamp'] = chunk.groupby('user_id')['timestamp'].shift(-1)
        chunk['visit_duration'] = (
            chunk['next_timestamp'] - chunk['timestamp']
        ).dt.total_seconds() / 60
        chunk['visit_duration'] = chunk['visit_duration'].clip(upper=180).fillna(30)
        
        return chunk
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to visit data."""
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['is_morning'] = data['hour'].between(6, 12)
        data['is_afternoon'] = data['hour'].between(12, 18)
        data['is_evening'] = data['hour'].between(18, 22)
        
        return data
    
    def _create_user_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create user preference profiles from historical visits."""
        user_profiles = data.groupby('user_id').agg({
            'poi_id': 'count',
            'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'general',
            'visit_duration': 'mean',
            'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else 12
        }).rename(columns={
            'poi_id': 'total_visits',
            'category': 'preferred_category',
            'visit_duration': 'avg_visit_duration',
            'hour': 'preferred_hour'
        })
        
        data = data.merge(user_profiles, on='user_id', how='left')
        return data
    
    def _cluster_locations(self, coordinates: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Cluster POI locations for spatial analysis."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(coordinates)
        return clusters
    
    def _get_season(self, month: int) -> str:
        """Map month to season."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _add_crowding_features(self, data: pd.DataFrame, crowding: pd.DataFrame) -> pd.DataFrame:
        """Add crowding features to visit data."""
        # Simplified merge based on POI and time window
        data['hour_window'] = data['timestamp'].dt.floor('H')
        crowding['hour_window'] = crowding['datetime'].dt.floor('H')
        
        data = data.merge(
            crowding[['poi_id', 'hour_window', 'crowding_level', 'crowding_ma_24h']],
            on=['poi_id', 'hour_window'],
            how='left'
        )
        
        data['crowding_level'] = data['crowding_level'].fillna(0.5)
        data['crowding_ma_24h'] = data['crowding_ma_24h'].fillna(0.5)
        
        return data
    
    def _add_contextual_features(self, data: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features to visit data."""
        data['date'] = data['timestamp'].dt.date
        context['date'] = context['date'].dt.date
        
        data = data.merge(
            context[['date', 'weather', 'temperature', 'is_holiday', 'season']],
            on='date',
            how='left'
        )
        
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different data dimensions."""
        # Category-weather interaction
        data['indoor_rainy'] = (
            (data['category'].isin(['museum', 'indoor'])) &
            (data['weather'] == 'rainy')
        ).astype(int)
        
        # Crowding-time interaction
        data['peak_hour_crowded'] = (
            (data['hour'].between(10, 16)) &
            (data['crowding_level'] > 0.7)
        ).astype(int)
        
        # User preference alignment
        data['preference_match'] = (
            data['category'] == data['preferred_category']
        ).astype(int)
        
        return data


class MetricsCalculator:
    """Comprehensive metrics calculation for tourism recommendation evaluation.
    
    This class implements various metrics for evaluating the performance of tourism
    recommendation systems, including sustainability measures, satisfaction metrics,
    and system efficiency indicators.
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Small constant for numerical stability
    
    def calculate_advanced_metrics(self, trajectories: Dict[int, List[Dict]], 
                                  poi_visits: Dict[int, int]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics for the tourism system.
        
        This method computes a wide range of metrics including sustainability indicators,
        satisfaction measures, and efficiency metrics to provide a holistic evaluation
        of system performance.
        """
        metrics = {}
        
        # Sustainability metrics
        metrics.update(self._calculate_sustainability_metrics(poi_visits))
        
        # Satisfaction metrics
        metrics.update(self._calculate_satisfaction_metrics(trajectories))
        
        # Efficiency metrics
        metrics.update(self._calculate_efficiency_metrics(trajectories))
        
        # Diversity metrics
        metrics.update(self._calculate_diversity_metrics(trajectories))
        
        # Fairness metrics
        metrics.update(self._calculate_fairness_metrics(trajectories, poi_visits))
        
        return metrics
    
    def _calculate_sustainability_metrics(self, poi_visits: Dict[int, int]) -> Dict[str, float]:
        """Calculate sustainability-related metrics."""
        visit_counts = list(poi_visits.values())
        total_visits = sum(visit_counts)
        
        if total_visits == 0:
            return {
                'gini_coefficient': 0.0,
                'entropy': 0.0,
                'coverage': 0.0
            }
        
        # Gini coefficient for visit distribution equality
        sorted_counts = sorted(visit_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Shannon entropy for visit distribution
        probabilities = np.array(visit_counts) / total_visits
        entropy = -np.sum(probabilities * np.log(probabilities + self.epsilon))
        
        # Coverage: percentage of POIs visited
        coverage = len([v for v in visit_counts if v > 0]) / len(poi_visits)
        
        # Concentration ratio (CR4: top 4 POIs' share)
        top_4_share = sum(sorted(visit_counts, reverse=True)[:4]) / total_visits if total_visits > 0 else 0
        
        return {
            'gini_coefficient': gini,
            'entropy': entropy,
            'coverage': coverage,
            'concentration_ratio': top_4_share
        }
    
    def _calculate_satisfaction_metrics(self, trajectories: Dict[int, List[Dict]]) -> Dict[str, float]:
        """Calculate tourist satisfaction metrics."""
        if not trajectories:
            return {
                'avg_satisfaction': 0.0,
                'satisfaction_std': 0.0,
                'completion_rate': 0.0
            }
        
        all_satisfactions = []
        completion_rates = []
        
        for agent_trajectory in trajectories.values():
            if agent_trajectory:
                satisfactions = [visit.get('satisfaction', 0) for visit in agent_trajectory]
                all_satisfactions.extend(satisfactions)
                
                # Completion rate: ratio of visited POIs to planned POIs
                completion_rates.append(len(agent_trajectory) / 10)  # Assuming 10 POIs planned
        
        avg_satisfaction = np.mean(all_satisfactions) if all_satisfactions else 0
        satisfaction_std = np.std(all_satisfactions) if all_satisfactions else 0
        avg_completion = np.mean(completion_rates) if completion_rates else 0
        
        # Calculate satisfaction trajectory (improvement over time)
        satisfaction_improvement = self._calculate_satisfaction_trend(trajectories)
        
        return {
            'avg_satisfaction': avg_satisfaction,
            'satisfaction_std': satisfaction_std,
            'completion_rate': avg_completion,
            'satisfaction_improvement': satisfaction_improvement
        }
    
    def _calculate_efficiency_metrics(self, trajectories: Dict[int, List[Dict]]) -> Dict[str, float]:
        """Calculate system efficiency metrics."""
        if not trajectories:
            return {
                'avg_waiting_time': 0.0,
                'time_utilization': 0.0,
                'travel_efficiency': 0.0
            }
        
        total_waiting = 0
        total_visits = 0
        total_travel_time = 0
        
        for agent_trajectory in trajectories.values():
            for visit in agent_trajectory:
                total_waiting += visit.get('waiting_time', 0)
                total_travel_time += visit.get('travel_time', 0)
                total_visits += 1
        
        avg_waiting = total_waiting / total_visits if total_visits > 0 else 0
        
        # Time utilization: ratio of productive time to total time
        total_time = 480 * len(trajectories)  # Assuming 8-hour days
        productive_time = total_time - total_waiting - total_travel_time
        time_utilization = productive_time / total_time if total_time > 0 else 0
        
        # Travel efficiency: inverse of average travel time per visit
        travel_efficiency = 1 / (1 + total_travel_time / total_visits) if total_visits > 0 else 0
        
        return {
            'avg_waiting_time': avg_waiting,
            'time_utilization': time_utilization,
            'travel_efficiency': travel_efficiency
        }
    
    def _calculate_diversity_metrics(self, trajectories: Dict[int, List[Dict]]) -> Dict[str, float]:
        """Calculate itinerary diversity metrics."""
        if not trajectories:
            return {
                'inter_agent_diversity': 0.0,
                'intra_agent_diversity': 0.0
            }
        
        # Inter-agent diversity: how different are agents' itineraries
        inter_diversity = self._calculate_inter_agent_diversity(trajectories)
        
        # Intra-agent diversity: variety within each agent's itinerary
        intra_diversity = self._calculate_intra_agent_diversity(trajectories)
        
        return {
            'inter_agent_diversity': inter_diversity,
            'intra_agent_diversity': intra_diversity
        }
    
    def _calculate_fairness_metrics(self, trajectories: Dict[int, List[Dict]], 
                                   poi_visits: Dict[int, int]) -> Dict[str, float]:
        """Calculate fairness metrics across agents and POIs."""
        # Agent fairness: equality of satisfaction across agents
        agent_satisfactions = []
        for agent_trajectory in trajectories.values():
            agent_satisfaction = sum(visit.get('satisfaction', 0) for visit in agent_trajectory)
            agent_satisfactions.append(agent_satisfaction)
        
        if agent_satisfactions:
            agent_fairness = 1 - self._calculate_gini(agent_satisfactions)
        else:
            agent_fairness = 0
        
        # POI fairness: equality of utilization across POIs
        poi_utilizations = list(poi_visits.values())
        if poi_utilizations:
            poi_fairness = 1 - self._calculate_gini(poi_utilizations)
        else:
            poi_fairness = 0
        
        return {
            'agent_fairness': agent_fairness,
            'poi_fairness': poi_fairness
        }
    
    def _calculate_satisfaction_trend(self, trajectories: Dict[int, List[Dict]]) -> float:
        """Calculate trend in satisfaction over time."""
        time_satisfactions = defaultdict(list)
        
        for agent_trajectory in trajectories.values():
            for i, visit in enumerate(agent_trajectory):
                time_satisfactions[i].append(visit.get('satisfaction', 0))
        
        if not time_satisfactions:
            return 0.0
        
        # Calculate linear trend
        times = sorted(time_satisfactions.keys())
        avg_satisfactions = [np.mean(time_satisfactions[t]) for t in times]
        
        if len(times) > 1:
            slope, _ = np.polyfit(times, avg_satisfactions, 1)
            return slope
        return 0.0
    
    def _calculate_inter_agent_diversity(self, trajectories: Dict[int, List[Dict]]) -> float:
        """Calculate diversity between different agents' itineraries."""
        if len(trajectories) < 2:
            return 0.0
        
        itineraries = []
        for agent_trajectory in trajectories.values():
            itinerary = [visit['poi_id'] for visit in agent_trajectory]
            itineraries.append(set(itinerary))
        
        # Calculate average Jaccard distance
        distances = []
        for i in range(len(itineraries)):
            for j in range(i + 1, len(itineraries)):
                intersection = len(itineraries[i] & itineraries[j])
                union = len(itineraries[i] | itineraries[j])
                jaccard = 1 - (intersection / union) if union > 0 else 0
                distances.append(jaccard)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_intra_agent_diversity(self, trajectories: Dict[int, List[Dict]]) -> float:
        """Calculate diversity within individual agents' itineraries."""
        diversities = []
        
        for agent_trajectory in trajectories.values():
            if len(agent_trajectory) > 1:
                categories = [visit.get('category', 'general') for visit in agent_trajectory]
                unique_categories = len(set(categories))
                diversity = unique_categories / len(categories)
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values."""
        if not values or sum(values) == 0:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


class Visualizer:
    """Visualization toolkit for generating publication-quality figures.
    
    This class provides comprehensive visualization capabilities for tourism data,
    including trajectory maps, metric dashboards, and statistical analyses.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[float, float] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 8)
        
    def generate_final_report(self, metrics: Dict[str, float], 
                            trajectories: Dict[int, List[Dict]]) -> None:
        """Generate comprehensive visualization report.
        
        This method creates a multi-panel figure containing all key visualizations
        for the tourism recommendation system evaluation.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Metrics dashboard
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metrics_dashboard(ax1, metrics)
        
        # Panel 2: Trajectory map
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_trajectory_map(ax2, trajectories)
        
        # Panel 3: Temporal patterns
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_temporal_patterns(ax3, trajectories)
        
        # Panel 4: Crowding distribution
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_crowding_distribution(ax4, trajectories)
        
        # Panel 5: Satisfaction analysis
        ax5 = fig.add_subplot(gs[0, 2])
        self._plot_satisfaction_analysis(ax5, trajectories)
        
        # Panel 6: POI utilization
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_poi_utilization(ax6, trajectories)
        
        # Panel 7: Efficiency breakdown
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_efficiency_breakdown(ax7, metrics)
        
        plt.suptitle('Tourism Recommendation System Evaluation Report', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'results/figures/evaluation_report_{timestamp}.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.savefig(f'results/figures/evaluation_report_{timestamp}.png', 
                   bbox_inches='tight', dpi=300)
        plt.show()
    
    def _plot_metrics_dashboard(self, ax: plt.Axes, metrics: Dict[str, float]) -> None:
        """Create metrics dashboard visualization."""
        # Organize metrics by category
        categories = {
            'Sustainability': ['gini_coefficient', 'entropy', 'coverage'],
            'Satisfaction': ['avg_satisfaction', 'satisfaction_std', 'completion_rate'],
            'Efficiency': ['time_utilization', 'travel_efficiency', 'avg_waiting_time'],
            'Fairness': ['agent_fairness', 'poi_fairness']
        }
        
        # Create grouped bar chart
        x_positions = []
        bar_values = []
        bar_labels = []
        colors = []
        
        position = 0
        for i, (category, metric_names) in enumerate(categories.items()):
            for metric_name in metric_names:
                if metric_name in metrics:
                    x_positions.append(position)
                    bar_values.append(metrics[metric_name])
                    bar_labels.append(metric_name.replace('_', ' ').title())
                    colors.append(self.color_palette[i % len(self.color_palette)])
                    position += 1
            position += 0.5  # Space between categories
        
        bars = ax.bar(x_positions, bar_values, color=colors)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        ax.set_ylabel('Metric Value')
        ax.set_title('System Performance Metrics Dashboard')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, bar_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_trajectory_map(self, ax: plt.Axes, trajectories: Dict[int, List[Dict]]) -> None:
        """Visualize agent trajectories on a map."""
        # Extract all POI positions (simplified for this example)
        poi_positions = {}
        for agent_trajectory in trajectories.values():
            for visit in agent_trajectory:
                poi_id = visit['poi_id']
                if poi_id not in poi_positions:
                    # Generate random positions for visualization
                    poi_positions[poi_id] = (
                        np.random.uniform(45.4, 45.5),
                        np.random.uniform(10.9, 11.0)
                    )
        
        # Plot trajectories
        for i, (agent_id, agent_trajectory) in enumerate(trajectories.items()):
            if len(agent_trajectory) > 1:
                trajectory_points = []
                for visit in agent_trajectory:
                    poi_id = visit['poi_id']
                    trajectory_points.append(poi_positions[poi_id])
                
                trajectory_points = np.array(trajectory_points)
                ax.plot(trajectory_points[:, 1], trajectory_points[:, 0],
                       'o-', alpha=0.5, linewidth=1, markersize=4,
                       color=self.color_palette[i % len(self.color_palette)],
                       label=f'Agent {agent_id}' if i < 5 else '')
        
        # Plot POI locations
        for poi_id, (lat, lon) in poi_positions.items():
            ax.scatter(lon, lat, s=100, c='red', marker='s', 
                      edgecolors='black', linewidth=1, zorder=5)
            ax.annotate(str(poi_id), (lon, lat), fontsize=8, ha='center')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Agent Trajectories Map')
        if len(trajectories) <= 5:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_temporal_patterns(self, ax: plt.Axes, trajectories: Dict[int, List[Dict]]) -> None:
        """Plot temporal visitation patterns."""
        hourly_visits = defaultdict(int)
        
        for agent_trajectory in trajectories.values():
            for i, visit in enumerate(agent_trajectory):
                # Estimate visit hour based on position in trajectory
                hour = 9 + i  # Starting at 9 AM
                hourly_visits[hour % 24] += 1
        
        hours = sorted(hourly_visits.keys())
        visits = [hourly_visits[h] for h in hours]
        
        ax.bar(hours, visits, color=self.color_palette[0])
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Visits')
        ax.set_title('Temporal Visitation Patterns')
        ax.set_xticks(range(9, 19))
        ax.grid(True, alpha=0.3)
    
    def _plot_crowding_distribution(self, ax: plt.Axes, trajectories: Dict[int, List[Dict]]) -> None:
        """Plot distribution of crowding levels encountered."""
        crowding_levels = []
        
        for agent_trajectory in trajectories.values():
            for visit in agent_trajectory:
                crowding = visit.get('crowding_level', np.random.random())
                crowding_levels.append(crowding)
        
        ax.hist(crowding_levels, bins=20, color=self.color_palette[1], 
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(crowding_levels), color='red', linestyle='--',
                  label=f'Mean: {np.mean(crowding_levels):.2f}')
        ax.set_xlabel('Crowding Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Crowding Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_satisfaction_analysis(self, ax: plt.Axes, trajectories: Dict[int, List[Dict]]) -> None:
        """Analyze satisfaction patterns."""
        agent_satisfactions = []
        
        for agent_trajectory in trajectories.values():
            total_satisfaction = sum(visit.get('satisfaction', 0) 
                                   for visit in agent_trajectory)
            agent_satisfactions.append(total_satisfaction)
        
        # Create box plot
        bp = ax.boxplot(agent_satisfactions, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor(self.color_palette[2])
        
        ax.set_ylabel('Total Satisfaction')
        ax.set_title('Agent Satisfaction Distribution')
        ax.set_xticklabels(['All Agents'])
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_sat = np.mean(agent_satisfactions)
        std_sat = np.std(agent_satisfactions)
        ax.text(0.7, mean_sat, f'μ={mean_sat:.2f}\nσ={std_sat:.2f}',
               transform=ax.get_yaxis_transform(), fontsize=10)
    
    def _plot_poi_utilization(self, ax: plt.Axes, trajectories: Dict[int, List[Dict]]) -> None:
        """Visualize POI utilization rates."""
        poi_visits = defaultdict(int)
        
        for agent_trajectory in trajectories.values():
            for visit in agent_trajectory:
                poi_visits[visit['poi_id']] += 1
        
        # Sort POIs by visit count
        sorted_pois = sorted(poi_visits.items(), key=lambda x: x[1], reverse=True)[:10]
        pois, visits = zip(*sorted_pois) if sorted_pois else ([], [])
        
        ax.barh(range(len(pois)), visits, color=self.color_palette[3])
        ax.set_yticks(range(len(pois)))
        ax.set_yticklabels([f'POI {p}' for p in pois])
        ax.set_xlabel('Number of Visits')
        ax.set_title('Top 10 POI Utilization')
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_breakdown(self, ax: plt.Axes, metrics: Dict[str, float]) -> None:
        """Create efficiency breakdown pie chart."""
        # Extract time components
        time_components = {
            'Productive Time': metrics.get('time_utilization', 0.6),
            'Waiting Time': 0.2,  # Estimated
            'Travel Time': 0.15,  # Estimated
            'Idle Time': 0.05  # Estimated
        }
        
        wedges, texts, autotexts = ax.pie(
            time_components.values(),
            labels=time_components.keys(),
            colors=self.color_palette[:len(time_components)],
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Time Efficiency Breakdown')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_id: Optional[int] = None) -> torch.device:
    """Get the appropriate compute device for PyTorch operations."""
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda')
    return torch.device('cpu')


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict[str, float], filepath: str) -> None:
    """Save model checkpoint with comprehensive state information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    
    # Also save in a versioned manner
    base_path = Path(filepath).parent
    filename = Path(filepath).stem
    versioned_path = base_path / f'{filename}_epoch_{epoch}.pt'
    torch.save(checkpoint, versioned_path)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load model checkpoint from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    return checkpoint


def calculate_trajectory_similarity(traj1: List[int], traj2: List[int]) -> float:
    """Calculate similarity between two POI trajectories using edit distance."""
    from difflib import SequenceMatcher
    
    matcher = SequenceMatcher(None, traj1, traj2)
    return matcher.ratio()


def optimize_trajectory_order(poi_list: List[int], distance_matrix: np.ndarray) -> List[int]:
    """Optimize POI visitation order using the traveling salesman problem solution."""
    n = len(poi_list)
    if n <= 2:
        return poi_list
    
    # Create cost matrix for selected POIs
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i, j] = distance_matrix[poi_list[i], poi_list[j]]
    
    # Solve using nearest neighbor heuristic (simplified)
    visited = [False] * n
    route = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        current = route[-1]
        next_poi = None
        min_dist = float('inf')
        
        for j in range(n):
            if not visited[j] and cost_matrix[current, j] < min_dist:
                min_dist = cost_matrix[current, j]
                next_poi = j
        
        if next_poi is not None:
            route.append(next_poi)
            visited[next_poi] = True
    
    return [poi_list[i] for i in route]


def generate_synthetic_trajectories(num_agents: int, num_pois: int, 
                                   trajectory_length: int) -> Dict[int, List[int]]:
    """Generate synthetic trajectories for testing and validation."""
    trajectories = {}
    
    for agent_id in range(num_agents):
        # Generate trajectory with some structure (nearby POIs more likely)
        trajectory = []
        current_poi = np.random.randint(0, num_pois)
        trajectory.append(current_poi)
        
        for _ in range(trajectory_length - 1):
            # Prefer nearby POIs (simplified proximity model)
            next_poi = (current_poi + np.random.randint(-2, 3)) % num_pois
            if next_poi not in trajectory:
                trajectory.append(next_poi)
                current_poi = next_poi
        
        trajectories[agent_id] = trajectory
    
    return trajectories