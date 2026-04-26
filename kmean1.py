import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# 1. Define the dataset
data = {
    'Point': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
    'x': [1, 2, 2, 1, 5, 6, 7, 8, 2, 7],
    'y': [2, 1, 4, 3, 8, 9, 8, 9, 2, 7]
}
df = pd.DataFrame(data)

# 2. Define Initial Centroids
c1 = np.array([1, 2])
c2 = np.array([8, 9])

# Function to assign clusters
def get_cluster(row):
    dist1 = np.sqrt((row['x'] - c1[0])**2 + (row['y'] - c1[1])**2)
    dist2 = np.sqrt((row['x'] - c2[0])**2 + (row['y'] - c2[1])**2)
    return 1 if dist1 < dist2 else 2

df['Cluster'] = df.apply(get_cluster, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(df['x'], df['y'], color='gray', s=100, label='Data Points')
ax1.set_title('Before K-Means')
ax1.set_xlabel('Visits (x)')
ax1.set_ylabel('Spend (y)')
ax1.grid(True, linestyle='--')
ax1.legend()

colors = {1: 'blue', 2: 'black'}

for cluster_id, color in colors.items():
    subset = df[df['Cluster'] == cluster_id]
    ax2.scatter(subset['x'], subset['y'], c=color, label=f'Cluster {cluster_id}', s=100)
    
    # Calculate circle center (using the cluster's mean)
    center_x = subset['x'].mean()
    center_y = subset['y'].mean()
    
    circle = patches.Circle((center_x, center_y), radius=2.5, 
                            edgecolor=color, facecolor=color, alpha=0.1, linestyle='--')
    ax2.add_patch(circle)

ax2.scatter([c1[0], c2[0]], [c1[1], c2[1]], color='red', s=200, label='Centroid', zorder=5, edgecolor='white')

ax2.set_title('After K-Means')
ax2.set_xlabel('Visits (x)')
ax2.set_ylabel('Spend (y)')
ax2.legend()
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()