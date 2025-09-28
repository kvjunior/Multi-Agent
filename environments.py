"""
environments.py - Multi-Agent Tourism Environment for Sustainable Itinerary Planning
Academic implementation for Information Technology & Tourism journal submission
Authors: [Your Name]
Institution: [Your Institution]
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.stats import norm
import heapq
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class POIInfo:
    """Data structure for Point of Interest information"""
    poi_id: int
    name: str
    latitude: float
    longitude: float
    category: str
    capacity: int
    visit_duration: float  # in minutes
    opening_hours: Tuple[int, int]  # (open_hour, close_hour)
    popularity_score: float
    sustainability_score: float
    entrance_fee: float


@dataclass
class AgentState:
    """State representation for individual tourist agent"""
    agent_id: int
    current_poi: Optional[int]
    visited_pois: Set[int]
    current_time: float
    remaining_budget: float
    satisfaction_score: float
    distance_traveled: float
    time_in_queues: float
    position: Tuple[float, float]  # (latitude, longitude)
    preference_vector: np.ndarray
    itinerary: List[int] = field(default_factory=list)


@dataclass
class EnvironmentContext:
    """Contextual information for the environment"""
    current_datetime: pd.Timestamp
    weather_condition: str  # 'sunny', 'rainy', 'cloudy'
    temperature: float
    is_holiday: bool
    is_weekend: bool
    season: str  # 'spring', 'summer', 'fall', 'winter'
    special_events: List[str] = field(default_factory=list)


class CrowdDynamicsModel:
    """Sophisticated crowd dynamics simulation based on queueing theory"""
    
    def __init__(self, poi_capacity: Dict[int, int], service_rates: Dict[int, float]):
        self.poi_capacity = poi_capacity
        self.service_rates = service_rates  # visitors per minute
        self.current_occupancy = defaultdict(int)
        self.queue_lengths = defaultdict(int)
        self.arrival_rates = defaultdict(float)
        self.historical_patterns = defaultdict(lambda: deque(maxlen=100))
        
    def update_occupancy(self, poi_id: int, delta: int):
        """Update POI occupancy with constraints"""
        self.current_occupancy[poi_id] = max(0, min(
            self.current_occupancy[poi_id] + delta,
            self.poi_capacity[poi_id]
        ))
        self.historical_patterns[poi_id].append(self.current_occupancy[poi_id])
    
    def get_waiting_time(self, poi_id: int, current_time: float) -> float:
        """Calculate expected waiting time using M/M/c queue model"""
        capacity = self.poi_capacity[poi_id]
        service_rate = self.service_rates[poi_id]
        current_queue = self.queue_lengths[poi_id]
        
        if current_queue == 0:
            return 0.0
        
        # M/M/c queue waiting time approximation
        utilization = self.current_occupancy[poi_id] / capacity
        if utilization >= 1.0:
            # System at capacity - use linear approximation
            return current_queue * (1.0 / service_rate)
        
        # Erlang C formula approximation
        rho = utilization
        c = int(capacity * 0.8)  # Effective servers
        
        if c == 0:
            return 0.0
            
        erlang_c = self._calculate_erlang_c(rho, c)
        waiting_time = erlang_c / (c * service_rate * (1 - rho))
        
        # Add stochastic component
        noise = np.random.normal(0, waiting_time * 0.1)
        return max(0, waiting_time + noise)
    
    def _calculate_erlang_c(self, rho: float, c: int) -> float:
        """Calculate Erlang C probability"""
        if rho >= 1:
            return 1.0
        
        a = rho * c
        sum_term = sum((a ** k) / np.math.factorial(k) for k in range(c))
        erlang_b = (a ** c) / (np.math.factorial(c) * (1 - rho))
        
        return erlang_b / (sum_term + erlang_b)
    
    def predict_future_crowding(self, poi_id: int, time_ahead: float) -> float:
        """Predict future crowding levels using historical patterns"""
        if len(self.historical_patterns[poi_id]) < 10:
            return self.current_occupancy[poi_id] / self.poi_capacity[poi_id]
        
        # Simple ARIMA-like prediction
        history = np.array(self.historical_patterns[poi_id])
        trend = np.polyfit(range(len(history)), history, deg=1)[0]
        
        predicted_occupancy = self.current_occupancy[poi_id] + trend * time_ahead
        predicted_occupancy = max(0, min(predicted_occupancy, self.poi_capacity[poi_id]))
        
        return predicted_occupancy / self.poi_capacity[poi_id]


class MultiAgentTourismEnv:
    """Multi-agent environment for sustainable tourism simulation"""
    
    def __init__(
        self,
        num_agents: int,
        poi_data: pd.DataFrame,
        distance_matrix: Optional[np.ndarray] = None,
        max_steps: int = 100,
        capacity_factor: float = 1.0,
        crowding_penalty: float = 0.3,
        sustainability_weight: float = 0.2,
        use_parallel: bool = True,
        num_workers: int = 4
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.capacity_factor = capacity_factor
        self.crowding_penalty = crowding_penalty
        self.sustainability_weight = sustainability_weight
        self.use_parallel = use_parallel
        self.num_workers = num_workers
        
        # Initialize POI information
        self.pois = self._initialize_pois(poi_data)
        self.num_pois = len(self.pois)
        self.poi_graph = self._build_poi_graph()
        
        # Distance matrix
        if distance_matrix is None:
            self.distance_matrix = self._compute_distance_matrix()
        else:
            self.distance_matrix = distance_matrix
        
        # Travel time matrix (assuming average speed)
        self.travel_time_matrix = self.distance_matrix / 4.0  # 4 km/h walking speed
        
        # Initialize crowd dynamics
        poi_capacities = {poi.poi_id: int(poi.capacity * capacity_factor) 
                         for poi in self.pois.values()}
        service_rates = {poi.poi_id: poi.capacity / poi.visit_duration 
                         for poi in self.pois.values()}
        self.crowd_dynamics = CrowdDynamicsModel(poi_capacities, service_rates)
        
        # Agent management
        self.agents = {}
        self.agent_trajectories = defaultdict(list)
        
        # Environment state
        self.current_step = 0
        self.context = None
        self.global_time = 0.0
        
        # Metrics tracking
        self.metrics = {
            'total_satisfaction': 0.0,
            'total_sustainability': 0.0,
            'crowding_violations': 0,
            'average_waiting_time': 0.0,
            'poi_utilization': defaultdict(float),
            'revenue_generated': 0.0
        }
        
        # Action and observation spaces
        self.action_space_size = self.num_pois + 1  # +1 for "wait" action
        self.observation_space_size = self._calculate_obs_space_size()
        
        # Parallel execution setup
        if self.use_parallel:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def _initialize_pois(self, poi_data: pd.DataFrame) -> Dict[int, POIInfo]:
        """Initialize POI information from dataframe"""
        pois = {}
        for idx, row in poi_data.iterrows():
            poi = POIInfo(
                poi_id=int(row['poi_id']),
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                category=row.get('category', 'general'),
                capacity=int(row['capacity']),
                visit_duration=row.get('visit_duration', 30.0),
                opening_hours=(row.get('open_hour', 9), row.get('close_hour', 18)),
                popularity_score=row.get('popularity', 0.5),
                sustainability_score=row.get('sustainability', 0.5),
                entrance_fee=row.get('entrance_fee', 0.0)
            )
            pois[poi.poi_id] = poi
        return pois
    
    def _build_poi_graph(self) -> nx.Graph:
        """Build graph representation of POI network"""
        G = nx.Graph()
        
        for poi_id, poi in self.pois.items():
            G.add_node(poi_id, 
                      pos=(poi.latitude, poi.longitude),
                      category=poi.category,
                      capacity=poi.capacity)
        
        # Add edges based on proximity and category similarity
        for poi1_id, poi1 in self.pois.items():
            for poi2_id, poi2 in self.pois.items():
                if poi1_id < poi2_id:
                    distance = self._calculate_distance(
                        (poi1.latitude, poi1.longitude),
                        (poi2.latitude, poi2.longitude)
                    )
                    
                    # Connect POIs within reasonable walking distance
                    if distance < 2.0:  # 2 km threshold
                        weight = distance
                        if poi1.category == poi2.category:
                            weight *= 0.8  # Prefer similar categories
                        G.add_edge(poi1_id, poi2_id, weight=weight)
        
        return G
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distances between POIs"""
        positions = np.array([[poi.latitude, poi.longitude] 
                              for poi in self.pois.values()])
        
        # Haversine distance for geographic coordinates
        lat_rad = np.radians(positions[:, 0])
        lon_rad = np.radians(positions[:, 1])
        
        lat_diff = lat_rad[:, np.newaxis] - lat_rad
        lon_diff = lon_rad[:, np.newaxis] - lon_rad
        
        a = np.sin(lat_diff/2)**2 + np.cos(lat_rad[:, np.newaxis]) * \
            np.cos(lat_rad) * np.sin(lon_diff/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        earth_radius = 6371  # km
        distances = earth_radius * c
        
        return distances
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    def _calculate_obs_space_size(self) -> int:
        """Calculate observation space dimensionality"""
        # Agent state: current POI, visited POIs, time, budget, position
        agent_state_dim = 1 + self.num_pois + 4
        
        # POI states: occupancy, queue length, predicted crowding
        poi_state_dim = self.num_pois * 3
        
        # Context: weather, time of day, day type
        context_dim = 10
        
        return agent_state_dim + poi_state_dim + context_dim
    
    def reset(self, context: Optional[EnvironmentContext] = None) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.global_time = 0.0
        
        # Set context
        if context is None:
            context = self._generate_random_context()
        self.context = context
        
        # Reset crowd dynamics
        self.crowd_dynamics = CrowdDynamicsModel(
            {poi.poi_id: int(poi.capacity * self.capacity_factor) 
             for poi in self.pois.values()},
            {poi.poi_id: poi.capacity / poi.visit_duration 
             for poi in self.pois.values()}
        )
        
        # Initialize agents
        self.agents = {}
        for i in range(self.num_agents):
            self.agents[i] = self._initialize_agent(i)
        
        # Clear trajectories
        self.agent_trajectories = defaultdict(list)
        
        # Reset metrics
        self.metrics = {
            'total_satisfaction': 0.0,
            'total_sustainability': 0.0,
            'crowding_violations': 0,
            'average_waiting_time': 0.0,
            'poi_utilization': defaultdict(float),
            'revenue_generated': 0.0
        }
        
        # Return initial observations
        return self._get_observations()
    
    def _generate_random_context(self) -> EnvironmentContext:
        """Generate random environmental context"""
        return EnvironmentContext(
            current_datetime=pd.Timestamp.now(),
            weather_condition=np.random.choice(['sunny', 'rainy', 'cloudy']),
            temperature=np.random.normal(20, 5),
            is_holiday=np.random.random() < 0.2,
            is_weekend=np.random.random() < 0.3,
            season=np.random.choice(['spring', 'summer', 'fall', 'winter']),
            special_events=[]
        )
    
    def _initialize_agent(self, agent_id: int) -> AgentState:
        """Initialize individual agent state"""
        # Random starting position (could be hotel or entry point)
        start_poi = np.random.choice(list(self.pois.keys()))
        start_pos = (self.pois[start_poi].latitude, self.pois[start_poi].longitude)
        
        # Generate preference vector based on agent type
        preference_vector = self._generate_preference_vector()
        
        return AgentState(
            agent_id=agent_id,
            current_poi=None,
            visited_pois=set(),
            current_time=0.0,
            remaining_budget=np.random.uniform(50, 200),
            satisfaction_score=0.0,
            distance_traveled=0.0,
            time_in_queues=0.0,
            position=start_pos,
            preference_vector=preference_vector,
            itinerary=[]
        )
    
    def _generate_preference_vector(self) -> np.ndarray:
        """Generate agent preference vector for different POI categories"""
        # Categories: cultural, nature, entertainment, dining, shopping
        preference_styles = [
            [0.8, 0.2, 0.3, 0.4, 0.1],  # Cultural enthusiast
            [0.2, 0.9, 0.1, 0.3, 0.0],  # Nature lover
            [0.3, 0.1, 0.9, 0.6, 0.4],  # Entertainment seeker
            [0.5, 0.5, 0.5, 0.5, 0.5],  # Balanced tourist
        ]
        
        style = preference_styles[np.random.choice(len(preference_styles))]
        # Add noise for individual variation
        noise = np.random.normal(0, 0.1, len(style))
        preferences = np.clip(np.array(style) + noise, 0, 1)
        
        return preferences / preferences.sum()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute one environment step for all agents"""
        assert len(actions) == self.num_agents, "Actions must be provided for all agents"
        
        # Process actions in parallel if configured
        if self.use_parallel:
            rewards, dones, infos = self._parallel_step(actions)
        else:
            rewards, dones, infos = self._sequential_step(actions)
        
        # Update global time
        self.global_time += 1.0
        self.current_step += 1
        
        # Update metrics
        self._update_global_metrics()
        
        # Check termination conditions
        if self.current_step >= self.max_steps:
            dones = np.ones(self.num_agents, dtype=bool)
        
        # Get new observations
        observations = self._get_observations()
        
        return observations, rewards, dones, infos
    
    def _parallel_step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Process agent actions in parallel"""
        futures = []
        for agent_id, action in enumerate(actions):
            future = self.executor.submit(self._process_agent_action, agent_id, action)
            futures.append(future)
        
        results = [future.result() for future in futures]
        
        rewards = np.array([r[0] for r in results])
        dones = np.array([r[1] for r in results])
        infos = {i: r[2] for i, r in enumerate(results)}
        
        return rewards, dones, infos
    
    def _sequential_step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Process agent actions sequentially"""
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents, dtype=bool)
        infos = {}
        
        for agent_id, action in enumerate(actions):
            reward, done, info = self._process_agent_action(agent_id, action)
            rewards[agent_id] = reward
            dones[agent_id] = done
            infos[agent_id] = info
        
        return rewards, dones, infos
    
    def _process_agent_action(self, agent_id: int, action: int) -> Tuple[float, bool, Dict]:
        """Process individual agent action"""
        agent = self.agents[agent_id]
        
        # Handle wait action
        if action == self.num_pois:
            agent.current_time += 10  # Wait 10 minutes
            return 0.0, False, {'action': 'wait'}
        
        # Check if action is valid
        if action < 0 or action >= self.num_pois:
            return -1.0, False, {'error': 'invalid_action'}
        
        target_poi_id = list(self.pois.keys())[action]
        
        # Check if POI is already visited
        if target_poi_id in agent.visited_pois:
            return -0.5, False, {'error': 'already_visited'}
        
        # Check if POI is open
        poi = self.pois[target_poi_id]
        current_hour = (agent.current_time / 60) % 24
        if not (poi.opening_hours[0] <= current_hour < poi.opening_hours[1]):
            return -0.3, False, {'error': 'poi_closed'}
        
        # Calculate travel time and cost
        if agent.current_poi is not None:
            current_idx = list(self.pois.keys()).index(agent.current_poi)
            target_idx = action
            travel_time = self.travel_time_matrix[current_idx, target_idx] * 60  # Convert to minutes
            travel_distance = self.distance_matrix[current_idx, target_idx]
        else:
            travel_time = 10  # Initial travel time
            travel_distance = 1.0
        
        # Calculate waiting time
        waiting_time = self.crowd_dynamics.get_waiting_time(target_poi_id, agent.current_time)
        
        # Check budget constraint
        if agent.remaining_budget < poi.entrance_fee:
            return -0.2, False, {'error': 'insufficient_budget'}
        
        # Update agent state
        agent.current_poi = target_poi_id
        agent.visited_pois.add(target_poi_id)
        agent.current_time += travel_time + waiting_time + poi.visit_duration
        agent.remaining_budget -= poi.entrance_fee
        agent.distance_traveled += travel_distance
        agent.time_in_queues += waiting_time
        agent.position = (poi.latitude, poi.longitude)
        agent.itinerary.append(target_poi_id)
        
        # Update crowd dynamics
        self.crowd_dynamics.update_occupancy(target_poi_id, 1)
        
        # Calculate reward
        reward = self._calculate_reward(agent, poi, waiting_time)
        
        # Update agent satisfaction
        agent.satisfaction_score += reward
        
        # Store trajectory
        self.agent_trajectories[agent_id].append({
            'poi_id': target_poi_id,
            'arrival_time': agent.current_time - poi.visit_duration - waiting_time,
            'waiting_time': waiting_time,
            'satisfaction': reward
        })
        
        # Check if agent is done (time limit or all POIs visited)
        done = agent.current_time > 480 or len(agent.visited_pois) == self.num_pois
        
        info = {
            'poi_visited': target_poi_id,
            'waiting_time': waiting_time,
            'travel_time': travel_time,
            'satisfaction': reward,
            'crowding_level': self.crowd_dynamics.current_occupancy[target_poi_id] / poi.capacity
        }
        
        return reward, done, info
    
    def _calculate_reward(self, agent: AgentState, poi: POIInfo, waiting_time: float) -> float:
        """Calculate reward for visiting a POI"""
        # Base reward from POI attractiveness and agent preferences
        category_idx = ['cultural', 'nature', 'entertainment', 'dining', 'shopping'].index(
            poi.category if poi.category in ['cultural', 'nature', 'entertainment', 'dining', 'shopping'] 
            else 'cultural'
        )
        preference_match = agent.preference_vector[category_idx]
        base_reward = poi.popularity_score * preference_match
        
        # Sustainability bonus
        sustainability_bonus = poi.sustainability_score * self.sustainability_weight
        
        # Crowding penalty
        crowding_level = self.crowd_dynamics.current_occupancy[poi.poi_id] / poi.capacity
        crowding_penalty = self.crowding_penalty * max(0, crowding_level - 0.7)
        
        # Waiting time penalty
        waiting_penalty = waiting_time / 60.0  # Normalize to hours
        
        # Time of day factor (prefer visiting during optimal hours)
        current_hour = (agent.current_time / 60) % 24
        if 10 <= current_hour <= 16:
            time_factor = 1.0
        elif 9 <= current_hour <= 17:
            time_factor = 0.8
        else:
            time_factor = 0.5
        
        # Weather factor
        weather_factor = 1.0
        if self.context.weather_condition == 'rainy' and poi.category == 'nature':
            weather_factor = 0.5
        elif self.context.weather_condition == 'sunny' and poi.category == 'nature':
            weather_factor = 1.2
        
        # Calculate final reward
        reward = (base_reward + sustainability_bonus) * time_factor * weather_factor
        reward -= crowding_penalty + waiting_penalty * 0.1
        
        return max(-1.0, min(1.0, reward))  # Clip to [-1, 1]
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents"""
        observations = np.zeros((self.num_agents, self.observation_space_size))
        
        for agent_id, agent in self.agents.items():
            obs = self._get_agent_observation(agent)
            observations[agent_id] = obs
        
        return observations
    
    def _get_agent_observation(self, agent: AgentState) -> np.ndarray:
        """Get observation for individual agent"""
        obs = []
        
        # Current POI (one-hot encoded)
        current_poi_encoding = np.zeros(self.num_pois)
        if agent.current_poi is not None:
            poi_idx = list(self.pois.keys()).index(agent.current_poi)
            current_poi_encoding[poi_idx] = 1
        obs.extend(current_poi_encoding)
        
        # Visited POIs (binary vector)
        visited_encoding = np.zeros(self.num_pois)
        for poi_id in agent.visited_pois:
            poi_idx = list(self.pois.keys()).index(poi_id)
            visited_encoding[poi_idx] = 1
        obs.extend(visited_encoding)
        
        # Agent state features
        obs.extend([
            agent.current_time / 480,  # Normalized time
            agent.remaining_budget / 200,  # Normalized budget
            agent.distance_traveled / 20,  # Normalized distance
            agent.satisfaction_score / 10  # Normalized satisfaction
        ])
        
        # POI occupancy levels
        for poi_id in self.pois.keys():
            occupancy = self.crowd_dynamics.current_occupancy[poi_id]
            capacity = self.pois[poi_id].capacity
            obs.append(occupancy / capacity if capacity > 0 else 0)
        
        # POI queue lengths
        for poi_id in self.pois.keys():
            queue_length = self.crowd_dynamics.queue_lengths[poi_id]
            obs.append(min(1.0, queue_length / 10))  # Normalized queue length
        
        # Predicted future crowding
        for poi_id in self.pois.keys():
            future_crowding = self.crowd_dynamics.predict_future_crowding(poi_id, 30)
            obs.append(future_crowding)
        
        # Context features
        obs.extend([
            1.0 if self.context.weather_condition == 'sunny' else 0.0,
            1.0 if self.context.weather_condition == 'rainy' else 0.0,
            1.0 if self.context.weather_condition == 'cloudy' else 0.0,
            self.context.temperature / 40,  # Normalized temperature
            1.0 if self.context.is_holiday else 0.0,
            1.0 if self.context.is_weekend else 0.0,
            1.0 if self.context.season == 'spring' else 0.0,
            1.0 if self.context.season == 'summer' else 0.0,
            1.0 if self.context.season == 'fall' else 0.0,
            1.0 if self.context.season == 'winter' else 0.0
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _update_global_metrics(self):
        """Update global environment metrics"""
        total_satisfaction = sum(agent.satisfaction_score for agent in self.agents.values())
        self.metrics['total_satisfaction'] = total_satisfaction / self.num_agents
        
        # Calculate sustainability score
        poi_visits = defaultdict(int)
        for agent_id, trajectory in self.agent_trajectories.items():
            for visit in trajectory:
                poi_visits[visit['poi_id']] += 1
        
        # Gini coefficient for visit distribution
        visit_counts = list(poi_visits.values())
        if visit_counts:
            visit_counts.sort()
            n = len(visit_counts)
            index = np.arange(1, n + 1)
            gini = (2 * index - n - 1).dot(visit_counts) / (n * sum(visit_counts))
            self.metrics['total_sustainability'] = 1 - gini  # Higher is more sustainable
        
        # Average waiting time
        total_waiting = sum(agent.time_in_queues for agent in self.agents.values())
        self.metrics['average_waiting_time'] = total_waiting / self.num_agents
        
        # POI utilization
        for poi_id, poi in self.pois.items():
            utilization = self.crowd_dynamics.current_occupancy[poi_id] / poi.capacity
            self.metrics['poi_utilization'][poi_id] = utilization
        
        # Revenue generated
        total_revenue = sum(poi.entrance_fee * poi_visits[poi.poi_id] 
                          for poi in self.pois.values())
        self.metrics['revenue_generated'] = total_revenue
    
    def get_trajectories(self) -> Dict[int, List[Dict]]:
        """Get all agent trajectories"""
        return dict(self.agent_trajectories)
    
    def get_poi_visits(self) -> Dict[int, int]:
        """Get visit counts for each POI"""
        poi_visits = defaultdict(int)
        for trajectory in self.agent_trajectories.values():
            for visit in trajectory:
                poi_visits[visit['poi_id']] += 1
        return dict(poi_visits)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current environment metrics"""
        return self.metrics.copy()
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment state"""
        if mode == 'human':
            self._print_state()
        elif mode == 'rgb_array':
            return self._generate_visualization()
        return None
    
    def _print_state(self):
        """Print current environment state"""
        print(f"\n--- Environment State (Step {self.current_step}) ---")
        print(f"Context: {self.context.weather_condition}, {self.context.temperature:.1f}°C")
        print(f"Time: {self.global_time:.1f} minutes")
        
        for agent_id, agent in self.agents.items():
            print(f"\nAgent {agent_id}:")
            print(f"  Current POI: {agent.current_poi}")
            print(f"  Visited: {agent.visited_pois}")
            print(f"  Satisfaction: {agent.satisfaction_score:.2f}")
            print(f"  Budget remaining: ${agent.remaining_budget:.2f}")
        
        print(f"\nMetrics:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
    
    def _generate_visualization(self) -> np.ndarray:
        """Generate RGB visualization of environment state"""
        # This would create a visual representation of the environment
        # For now, returning a placeholder
        return np.zeros((600, 800, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def create_verona_environment(config: Dict[str, Any]) -> MultiAgentTourismEnv:
    """Factory function to create Verona tourism environment"""
    # Load Verona POI data
    poi_data = pd.DataFrame({
        'poi_id': [42, 49, 52, 54, 58, 59, 61, 62, 63, 71, 75, 76, 201, 202, 300, 301, 302, 303],
        'name': ['Archaeological Museum', 'Arena Amphitheatre', 'The Cathedral', 
                'Church of St. Anastasia', 'Palazzo della Ragione', 'Lamberti Tower',
                'Juliets House', 'Church of St. Fermo', 'Church of St. Zeno',
                'Castelvecchio Museum', 'Giustis Garden', 'The Maffeiano Museum',
                'Natural History Museum', 'Frescoes Museum', 'Miniscalchi Museum',
                'Palazzo Maffei', 'National Museum', 'Eataly Verona'],
        'latitude': [45.4384, 45.4391, 45.4468, 45.4455, 45.4415, 45.4414, 
                     45.4420, 45.4384, 45.4384, 45.4397, 45.4435, 45.4388,
                     45.4457, 45.4393, 45.4470, 45.4424, 45.4456, 45.4330],
        'longitude': [10.9916, 10.9944, 10.9963, 10.9989, 10.9945, 10.9943,
                      10.9973, 10.9910, 10.9797, 10.9873, 11.0042, 10.9918,
                      10.9977, 10.9900, 10.9951, 10.9965, 10.9975, 10.9850],
        'category': ['cultural'] * 18,  # Simplified for this example
        'capacity': [100, 500, 200, 150, 80, 60, 300, 100, 120, 250, 50, 80,
                    60, 100, 40, 150, 70, 200],
        'visit_duration': [45, 90, 30, 30, 20, 20, 20, 30, 45, 60, 30, 30,
                          45, 30, 30, 45, 45, 60],
        'popularity': [0.6, 0.95, 0.7, 0.75, 0.5, 0.8, 0.9, 0.6, 0.65, 0.8,
                       0.4, 0.5, 0.3, 0.6, 0.2, 0.5, 0.3, 0.4],
        'sustainability': [0.7, 0.3, 0.8, 0.8, 0.9, 0.6, 0.4, 0.8, 0.9, 0.6,
                          0.9, 0.8, 0.9, 0.7, 0.9, 0.6, 0.8, 0.5],
        'entrance_fee': [10, 15, 5, 5, 8, 8, 6, 5, 5, 10, 10, 8, 5, 8, 6, 10, 8, 0]
    })
    
    # Add opening hours
    poi_data['open_hour'] = 9
    poi_data['close_hour'] = 18
    
    return MultiAgentTourismEnv(
        num_agents=config.get('num_agents', 10),
        poi_data=poi_data,
        max_steps=config.get('max_steps', 100),
        capacity_factor=config.get('capacity_factor', 1.0),
        crowding_penalty=config.get('crowding_penalty', 0.3),
        sustainability_weight=config.get('sustainability_weight', 0.2),
        use_parallel=config.get('use_parallel', True),
        num_workers=config.get('num_workers', 4)
    )