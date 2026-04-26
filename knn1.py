import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the historical dataset 
data = {
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'],
    'Attendance': [95, 55, 88, 60, 92, 45, 98, 50, 85, 65],
    'Quiz': [92, 45, 85, 50, 88, 40, 95, 55, 80, 60],
    'Class': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail']
}
df = pd.DataFrame(data)

# 2. Define the new entry
new_x, new_y = 80, 75
k = 5

# 3. SOLVE: Calculate Euclidean distance 
df['Distance'] = np.sqrt((df['Attendance'] - new_x)**2 + (df['Quiz'] - new_y)**2)

# 4. SOLVE: Sort in Ascending Order
df_sorted = df.sort_values(by='Distance')

# 5. SOLVE: Get Majority Vote 
nearest_neighbors = df_sorted.head(k)
predicted_class = nearest_neighbors['Class'].mode()[0]

print(f"Algorithm solved. Predicted class for ({new_x}, {new_y}) is: {predicted_class}")

# 6. PLOT: Visualize Before and After
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_setup(ax, title):
    for status, color in [('Pass', 'green'), ('Fail', 'orange')]:
        subset = df[df['Class'] == status]
        ax.scatter(subset['Attendance'], subset['Quiz'], label=f'Class: {status}', color=color, s=100, alpha=0.6)
    ax.set_xlabel('Attendance (X1)')
    ax.set_ylabel('Quiz (X2)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

# Before K-NN (Unclassified)
plot_setup(ax1, 'Before K-NN')
ax1.scatter(new_x, new_y, color='blue', label='New Data Point', s=150, edgecolors='black', marker='D')
ax1.legend()

# After K-NN (Classified by algorithm result)
plot_setup(ax2, 'After K-NN')
# Dynamically pick color based on the solver result
result_color = 'green' if predicted_class == 'Pass' else 'orange'
ax2.scatter(new_x, new_y, color=result_color, label=f'Assigned: {predicted_class}', s=150, edgecolors='black', marker='*')
ax2.legend()

plt.tight_layout()
plt.show()