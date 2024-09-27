import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

# Generate sample data
np.random.seed(42)
n_samples = 1000
age = np.random.randint(18, 80, n_samples)
income = np.random.randint(20000, 200000, n_samples)
spending = np.random.randint(1000, 50000, n_samples)
data = np.column_stack((age, income, spending))

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

class SOM:
    def __init__(self, x, y, input_dim, learning_rate=0.1, radius=None, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(x, y) / 2
        self.weights = np.random.rand(x, y, input_dim)

    def get_bmu(self, sample):
        distances = np.sum((self.weights - sample) ** 2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_weights(self, sample, bmu, iteration):
        learning_rate = self.learning_rate * np.exp(-iteration / 1000)
        radius = self.radius * np.exp(-iteration / 1000)
        
        for x in range(self.x):
            for y in range(self.y):
                distance = np.sum((np.array([x, y]) - np.array(bmu)) ** 2)
                if distance <= radius ** 2:
                    influence = np.exp(-distance / (2 * (radius ** 2)))
                    self.weights[x, y] += learning_rate * influence * (sample - self.weights[x, y])

    def train(self, data, iterations):
        for i in range(iterations):
            sample = data[np.random.randint(len(data))]
            bmu = self.get_bmu(sample)
            self.update_weights(sample, bmu, i)

# Create and train the SOM
som = SOM(10, 10, 3, learning_rate=0.1, radius=5, random_seed=42)
som.train(data_normalized, 10000)

# Visualize the results
fig, ax = plt.subplots(figsize=(12, 10))

# Create a custom colormap
colors = ['blue', 'green', 'yellow', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Plot each data point
for i, sample in enumerate(data_normalized):
    bmu = som.get_bmu(sample)
    ax.plot(bmu[0], bmu[1], 'o', markersize=6, 
            color=cmap(sample[0]), alpha=0.7)

ax.set_title('Customer Segmentation using SOM', fontsize=16)
ax.set_xlabel('SOM X', fontsize=12)
ax.set_ylabel('SOM Y', fontsize=12)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Normalized Age', orientation='vertical')
cbar.set_label('Normalized Age', fontsize=12)

plt.tight_layout()
plt.show()

# Analyze a few samples
print("Sample customer segments:")
for i in range(5):
    sample = data_normalized[i]
    bmu = som.get_bmu(sample)
    original_data = scaler.inverse_transform([sample])[0]
    print(f"Customer {i+1}: Age: {original_data[0]:.0f}, Income: ${original_data[1]:.0f}, "
          f"Spending: ${original_data[2]:.0f} - Segment: {bmu}")