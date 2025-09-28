import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.interpolate import make_interp_spline
from scipy.stats import multivariate_normal
import matplotlib.patches as mpatches

# Set publication-quality parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.linewidth'] = 0.3
plt.rcParams['lines.linewidth'] = 1.0

# Define consistent color palette for tourist archetypes
archetype_colors = {
    'Cultural Enthusiast': '#2E86AB',
    'Entertainment Seeker': '#A23B72',
    'Nature Lover': '#59A14F',
    'Family Tourist': '#F18F01',
    'Budget Traveler': '#C73E1D'
}

def generate_policy_embeddings(n_agents=500, n_features=50):
    """Generate synthetic policy embeddings for t-SNE visualization"""
    np.random.seed(42)
    
    # Create clustered data for each archetype
    embeddings = []
    labels = []
    archetypes = list(archetype_colors.keys())
    
    for i, archetype in enumerate(archetypes):
        n_cluster = n_agents // len(archetypes)
        
        # Create cluster center with some separation
        center = np.random.randn(n_features) * 3
        center[i*10:(i+1)*10] *= 2  # Make certain features more prominent
        
        # Generate points around center
        cluster_data = center + np.random.randn(n_cluster, n_features) * 0.5
        embeddings.append(cluster_data)
        labels.extend([archetype] * n_cluster)
    
    embeddings = np.vstack(embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(embeddings)
    
    return embedded, labels

def generate_temporal_patterns():
    """Generate temporal coordination patterns for different agent types"""
    hours = np.arange(9, 19)  # 9 AM to 6 PM
    
    patterns = {}
    
    # Cultural Enthusiast: morning preference
    cultural = np.array([0.25, 0.30, 0.28, 0.20, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02])
    
    # Entertainment Seeker: afternoon preference
    entertainment = np.array([0.05, 0.08, 0.10, 0.15, 0.22, 0.28, 0.30, 0.25, 0.18, 0.12])
    
    # Nature Lover: mid-morning to early afternoon
    nature = np.array([0.15, 0.22, 0.25, 0.23, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04])
    
    # Family Tourist: mid-day concentration
    family = np.array([0.08, 0.12, 0.18, 0.25, 0.28, 0.22, 0.15, 0.10, 0.08, 0.05])
    
    # Budget Traveler: distributed
    budget = np.array([0.12, 0.13, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07])
    
    # Normalize
    patterns['Cultural Enthusiast'] = cultural / cultural.sum()
    patterns['Entertainment Seeker'] = entertainment / entertainment.sum()
    patterns['Nature Lover'] = nature / nature.sum()
    patterns['Family Tourist'] = family / family.sum()
    patterns['Budget Traveler'] = budget / budget.sum()
    
    return hours, patterns

def generate_spatial_territories():
    """Generate spatial zone preferences for agent groups"""
    zones = ['Historical', 'Entertainment', 'Nature', 'Shopping', 'Dining']
    
    # Create preference matrix (archetypes x zones)
    preferences = np.array([
        [0.45, 0.15, 0.10, 0.15, 0.15],  # Cultural Enthusiast
        [0.10, 0.40, 0.10, 0.20, 0.20],  # Entertainment Seeker
        [0.15, 0.10, 0.45, 0.10, 0.20],  # Nature Lover
        [0.20, 0.25, 0.20, 0.20, 0.15],  # Family Tourist
        [0.15, 0.15, 0.15, 0.30, 0.25],  # Budget Traveler
    ])
    
    # Add some noise for realism
    preferences += np.random.normal(0, 0.02, preferences.shape)
    preferences = np.clip(preferences, 0, 1)
    
    # Normalize rows
    preferences = preferences / preferences.sum(axis=1, keepdims=True)
    
    return zones, preferences

def generate_entropy_evolution():
    """Generate entropy evolution during training"""
    episodes = np.linspace(0, 8500, 100)
    
    # Create realistic entropy curve with phases
    entropy = []
    for ep in episodes:
        if ep < 1000:  # Initial phase
            val = 2.3 - 0.0005 * ep + np.random.normal(0, 0.05)
        elif ep < 4000:  # Transition phase
            val = 1.8 - 0.00013 * (ep - 1000) + np.random.normal(0, 0.03)
        elif ep < 7000:  # Spatial coordination phase
            val = 1.4 + 0.00010 * (ep - 4000) + np.random.normal(0, 0.02)
        else:  # Convergence phase
            val = 1.7 + np.random.normal(0, 0.01)
        entropy.append(val)
    
    entropy = np.array(entropy)
    
    # Smooth the curve
    from scipy.ndimage import gaussian_filter1d
    entropy_smooth = gaussian_filter1d(entropy, sigma=2)
    
    return episodes, entropy, entropy_smooth

# Create the figure
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.3)

# Panel (a): t-SNE visualization
ax1 = fig.add_subplot(gs[0, 0])
embedded, labels = generate_policy_embeddings()

for archetype in archetype_colors.keys():
    mask = np.array(labels) == archetype
    ax1.scatter(embedded[mask, 0], embedded[mask, 1], 
               c=archetype_colors[archetype], label=archetype,
               alpha=0.6, s=10, edgecolors='white', linewidth=0.5)

ax1.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax1.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax1.set_title('(a) Policy Embedding Clusters', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fancybox=False, 
          edgecolor='black', framealpha=0.9, fontsize=7)
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
ax1.set_xlim(embedded[:, 0].min() - 5, embedded[:, 0].max() + 5)
ax1.set_ylim(embedded[:, 1].min() - 5, embedded[:, 1].max() + 5)

# Panel (b): Temporal coordination patterns
ax2 = fig.add_subplot(gs[0, 1])
hours, patterns = generate_temporal_patterns()

x_smooth = np.linspace(hours.min(), hours.max(), 300)
for archetype, pattern in patterns.items():
    # Create smooth interpolation
    spline = make_interp_spline(hours, pattern, k=3)
    y_smooth = spline(x_smooth)
    ax2.plot(x_smooth, y_smooth, label=archetype, 
            color=archetype_colors[archetype], linewidth=1.5, alpha=0.8)
    ax2.fill_between(x_smooth, 0, y_smooth, 
                     color=archetype_colors[archetype], alpha=0.1)

ax2.set_xlabel('Hour of Day', fontsize=10)
ax2.set_ylabel('POI Selection Probability', fontsize=10)
ax2.set_title('(b) Temporal Coordination Patterns', fontsize=11, fontweight='bold')
ax2.set_xticks(hours)
ax2.set_xticklabels([f'{h}:00' for h in hours], rotation=45, ha='right')
ax2.legend(loc='upper right', frameon=True, fancybox=False, 
          edgecolor='black', framealpha=0.9, fontsize=7)
ax2.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.3)
ax2.set_ylim(0, 0.35)

# Panel (c): Spatial territory emergence
ax3 = fig.add_subplot(gs[1, 0])
zones, preferences = generate_spatial_territories()

im = ax3.imshow(preferences, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
ax3.set_xticks(range(len(zones)))
ax3.set_xticklabels(zones, rotation=45, ha='right')
ax3.set_yticks(range(len(archetype_colors)))
ax3.set_yticklabels(list(archetype_colors.keys()))
ax3.set_xlabel('Zone', fontsize=10)
ax3.set_ylabel('Tourist Archetype', fontsize=10)
ax3.set_title('(c) Spatial Territory Preferences', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(preferences.shape[0]):
    for j in range(preferences.shape[1]):
        text = ax3.text(j, i, f'{preferences[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=7)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('Preference Strength', rotation=270, labelpad=15, fontsize=9)
cbar.ax.tick_params(labelsize=7)

# Panel (d): Action entropy evolution
ax4 = fig.add_subplot(gs[1, 1])
episodes, entropy_raw, entropy_smooth = generate_entropy_evolution()

# Plot raw and smoothed entropy
ax4.plot(episodes, entropy_raw, alpha=0.3, color='gray', linewidth=0.5, label='Raw')
ax4.plot(episodes, entropy_smooth, color='darkblue', linewidth=2, label='Smoothed')

# Add phase boundaries
phase_boundaries = [1000, 4000, 7000]
phase_names = ['Initial', 'Transition', 'Spatial', 'Convergence']
colors_phases = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']

for i, boundary in enumerate(phase_boundaries + [8500]):
    start = 0 if i == 0 else phase_boundaries[i-1]
    ax4.axvspan(start, boundary, alpha=0.15, color=colors_phases[i])
    mid_point = (start + boundary) / 2
    ax4.text(mid_point, 2.4, phase_names[i], ha='center', 
            fontsize=8, style='italic')

ax4.axvline(x=1000, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
ax4.axvline(x=4000, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
ax4.axvline(x=7000, color='black', linestyle='--', alpha=0.3, linewidth=0.5)

ax4.set_xlabel('Training Episodes', fontsize=10)
ax4.set_ylabel('Action Entropy (bits)', fontsize=10)
ax4.set_title('(d) Entropy Evolution During Training', fontsize=11, fontweight='bold')
ax4.legend(loc='upper right', frameon=True, fancybox=False, 
          edgecolor='black', framealpha=0.9, fontsize=7)
ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
ax4.set_xlim(0, 8500)
ax4.set_ylim(1.2, 2.5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('coordination_patterns.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('coordination_patterns.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

print("Figure generated successfully!")
print("Saved as 'figures/coordination_patterns.pdf' and 'coordination_patterns.png'")