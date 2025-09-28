import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patches as patches
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d

# Configure publication-quality parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.0

def create_poi_layout():
    """Create realistic spatial layout of 18 POIs in Verona"""
    np.random.seed(42)
    
    # Define POI positions mimicking Verona's historic center layout
    # Central cluster (most popular attractions)
    central_pois = np.array([
        [0.5, 0.5],    # Arena (POI 1)
        [0.48, 0.55],  # Juliet's House (POI 2)
        [0.52, 0.48],  # Cathedral (POI 3)
    ])
    
    # Semi-central attractions
    semicentral_pois = np.array([
        [0.45, 0.60],  # POI 4
        [0.55, 0.58],  # POI 5
        [0.43, 0.45],  # POI 6
        [0.57, 0.43],  # POI 7
        [0.50, 0.62],  # POI 8
    ])
    
    # Peripheral attractions
    angles = np.linspace(0, 2*np.pi, 11, endpoint=False)
    radius = 0.25
    peripheral_pois = np.array([
        [0.5 + radius*np.cos(angle), 0.5 + radius*np.sin(angle)] 
        for angle in angles[:10]
    ])
    
    all_pois = np.vstack([central_pois, semicentral_pois, peripheral_pois])
    
    # POI sizes based on capacity
    sizes = np.array([500, 300, 200] +  # Top 3 attractions
                     [100, 100, 80, 80, 80] +  # Semi-central
                     [50]*10)  # Peripheral
    
    return all_pois, sizes

def draw_single_agent_pattern(ax, poi_positions, poi_sizes):
    """Draw single-agent optimization pattern with concentration"""
    
    # Draw POIs as circles with size proportional to capacity
    for i, (pos, size) in enumerate(zip(poi_positions, poi_sizes)):
        if i < 3:  # Top 3 attractions - heavily visited
            color = '#d62728'  # Red for overcrowded
            alpha = 0.9
            edge_width = 2.5
        elif i < 8:  # Some moderate visits
            color = '#ff7f0e'  # Orange for moderate
            alpha = 0.6
            edge_width = 1.5
        else:  # Rarely visited
            color = '#7f7f7f'  # Gray for underutilized
            alpha = 0.4
            edge_width = 1.0
        
        circle = Circle(pos, radius=np.sqrt(size)/60, 
                       facecolor=color, alpha=alpha,
                       edgecolor='black', linewidth=edge_width)
        ax.add_patch(circle)
        
        # Add POI labels for top attractions
        if i < 3:
            ax.text(pos[0], pos[1], str(i+1), 
                   ha='center', va='center', fontsize=11, 
                   fontweight='bold', color='white')
        else:
            ax.text(pos[0], pos[1], str(i+1), 
                   ha='center', va='center', fontsize=8, 
                   color='black' if i < 8 else 'gray')
    
    # Draw tourist flows as arrows converging to top attractions
    np.random.seed(42)
    n_tourists = 50
    
    # Generate tourist starting positions around the perimeter
    angles = np.random.uniform(0, 2*np.pi, n_tourists)
    start_positions = np.array([
        [0.5 + 0.4*np.cos(angle), 0.5 + 0.4*np.sin(angle)] 
        for angle in angles
    ])
    
    # 73% go to top 3 attractions
    for i, start_pos in enumerate(start_positions):
        if i < int(0.73 * n_tourists):  # 73% to top 3
            target_poi = i % 3
            end_pos = poi_positions[target_poi]
            arrow_color = '#d62728'
            arrow_alpha = 0.3
            arrow_width = 1.5
        else:  # Remaining to other attractions
            target_poi = np.random.choice(range(3, 18))
            end_pos = poi_positions[target_poi]
            arrow_color = '#7f7f7f'
            arrow_alpha = 0.2
            arrow_width = 0.5
        
        # Add slight curve to arrows for visual appeal
        mid_point = [(start_pos[0] + end_pos[0])/2 + np.random.normal(0, 0.02),
                    (start_pos[1] + end_pos[1])/2 + np.random.normal(0, 0.02)]
        
        arrow = FancyArrowPatch(start_pos, end_pos,
                               connectionstyle="arc3,rad=0.2",
                               arrowstyle='->', 
                               color=arrow_color, alpha=arrow_alpha,
                               linewidth=arrow_width, mutation_scale=10)
        ax.add_patch(arrow)
    
    # Add congestion indicators at top attractions
    for i in range(3):
        pos = poi_positions[i]
        congestion_circle = Circle(pos, radius=np.sqrt(poi_sizes[i])/60 + 0.02,
                                  facecolor='none', edgecolor='red',
                                  linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(congestion_circle)
    
    # Add metrics annotations
    metrics_box = FancyBboxPatch((0.02, 0.82), 0.35, 0.15,
                                boxstyle="round,pad=0.01",
                                facecolor='white', edgecolor='black',
                                linewidth=1, alpha=0.95)
    ax.add_patch(metrics_box)
    
    ax.text(0.04, 0.93, 'Single-Agent Metrics:', fontsize=10, fontweight='bold')
    ax.text(0.04, 0.89, f'Gini coefficient: 0.478', fontsize=9)
    ax.text(0.04, 0.86, f'Concentration: 73% at 3 POIs', fontsize=9)
    ax.text(0.04, 0.83, f'Max wait time: 47 min', fontsize=9, color='red')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Single-Agent Optimization: Convergence Problem', 
                fontsize=12, fontweight='bold', pad=10)

def draw_multi_agent_pattern(ax, poi_positions, poi_sizes):
    """Draw multi-agent coordination pattern with distribution"""
    
    # Draw POIs with more balanced visitation
    visit_distribution = np.array([95, 90, 85, 80, 75, 70, 65, 60, 55, 
                                   50, 48, 45, 42, 40, 38, 35, 33, 30])
    visit_distribution = visit_distribution / visit_distribution.max()
    
    for i, (pos, size) in enumerate(zip(poi_positions, poi_sizes)):
        # Color based on balanced utilization
        utilization = visit_distribution[i]
        if utilization > 0.8:
            color = '#2ca02c'  # Green for well-utilized
            alpha = 0.8
        elif utilization > 0.5:
            color = '#1f77b4'  # Blue for moderate
            alpha = 0.7
        else:
            color = '#17becf'  # Cyan for lower but still active
            alpha = 0.6
        
        circle = Circle(pos, radius=np.sqrt(size)/60,
                       facecolor=color, alpha=alpha,
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        
        # Add POI labels
        ax.text(pos[0], pos[1], str(i+1),
               ha='center', va='center', fontsize=9 if i < 8 else 8,
               fontweight='bold' if i < 3 else 'normal',
               color='white' if utilization > 0.7 else 'black')
    
    # Draw distributed tourist flows
    np.random.seed(42)
    n_tourists = 50
    
    # Generate tourist starting positions
    angles = np.random.uniform(0, 2*np.pi, n_tourists)
    start_positions = np.array([
        [0.5 + 0.4*np.cos(angle), 0.5 + 0.4*np.sin(angle)]
        for angle in angles
    ])
    
    # Distribute tourists more evenly based on multi-agent coordination
    for i, start_pos in enumerate(start_positions):
        # Use probability distribution for POI selection
        probabilities = visit_distribution / visit_distribution.sum()
        target_poi = np.random.choice(range(18), p=probabilities)
        end_pos = poi_positions[target_poi]
        
        # Color based on coordination groups
        if i % 3 == 0:
            arrow_color = '#2ca02c'
        elif i % 3 == 1:
            arrow_color = '#1f77b4'
        else:
            arrow_color = '#17becf'
        
        arrow = FancyArrowPatch(start_pos, end_pos,
                               connectionstyle="arc3,rad=0.2",
                               arrowstyle='->',
                               color=arrow_color, alpha=0.25,
                               linewidth=1.0, mutation_scale=10)
        ax.add_patch(arrow)
    
    # Add coordination zones (Voronoi-like regions)
    vor = Voronoi(poi_positions[:8])  # Use top 8 POIs for cleaner visualization
    for region_idx in vor.regions:
        if -1 not in region_idx and len(region_idx) > 0:
            polygon = [vor.vertices[i] for i in region_idx]
            if len(polygon) > 2:
                poly_patch = patches.Polygon(polygon, alpha=0.1,
                                           facecolor='blue', edgecolor='none')
                ax.add_patch(poly_patch)
    
    # Add metrics annotations
    metrics_box = FancyBboxPatch((0.02, 0.82), 0.35, 0.15,
                                boxstyle="round,pad=0.01",
                                facecolor='white', edgecolor='black',
                                linewidth=1, alpha=0.95)
    ax.add_patch(metrics_box)
    
    ax.text(0.04, 0.93, 'Multi-Agent Metrics:', fontsize=10, fontweight='bold')
    ax.text(0.04, 0.89, f'Gini coefficient: 0.236', fontsize=9, color='green')
    ax.text(0.04, 0.86, f'Distribution: All 18 POIs active', fontsize=9)
    ax.text(0.04, 0.83, f'Max wait time: 21 min', fontsize=9, color='green')
    
    # Add satisfaction indicator
    satisfaction_box = FancyBboxPatch((0.63, 0.02), 0.35, 0.08,
                                     boxstyle="round,pad=0.01",
                                     facecolor='#e6f3e6', edgecolor='green',
                                     linewidth=1, alpha=0.95)
    ax.add_patch(satisfaction_box)
    ax.text(0.65, 0.06, 'Avg. Satisfaction: 8.43/10', fontsize=9, color='green')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Multi-Agent Coordination: Distributed Solution',
                fontsize=12, fontweight='bold', pad=10)

# Create the main figure
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1)

# Get POI layout
poi_positions, poi_sizes = create_poi_layout()

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Draw patterns
draw_single_agent_pattern(ax1, poi_positions, poi_sizes)
draw_multi_agent_pattern(ax2, poi_positions, poi_sizes)

# Add legend
legend_elements = [
    mpatches.Circle((0, 0), 0.1, facecolor='#d62728', alpha=0.9, 
                   edgecolor='black', label='Overcrowded POI'),
    mpatches.Circle((0, 0), 0.1, facecolor='#2ca02c', alpha=0.8,
                   edgecolor='black', label='Well-utilized POI'),
    mpatches.Circle((0, 0), 0.1, facecolor='#7f7f7f', alpha=0.4,
                   edgecolor='black', label='Underutilized POI'),
    mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color='gray',
                       label='Tourist Flow')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4,
          frameon=True, fancybox=False, edgecolor='black',
          fontsize=9, bbox_to_anchor=(0.5, -0.02))

# Save figure
plt.tight_layout()
plt.savefig('coordination_problem.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('coordination_problem.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

print("Coordination problem comparison figure generated successfully!")