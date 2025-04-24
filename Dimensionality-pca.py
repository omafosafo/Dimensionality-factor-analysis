import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

# Set random seed for reproducibility
np.random.seed(42)

# Generate latent factors
n_samples = 10000
f1 = np.random.normal(0, 1, n_samples)
f2 = np.random.normal(0, 1, n_samples)

# Generate observed features with noise
epsilon = np.random.normal(0, 1, (n_samples, 5))
x1 = 2*f1 + 3*f2 + epsilon[:, 0]
x2 = f1 - 10*f2 + epsilon[:, 1]
x3 = -5*f1 + 5*f2 + epsilon[:, 2]
x4 = 7*f1 + 11*f2 + epsilon[:, 3]
x5 = -6*f1 - 7*f2 + epsilon[:, 4]

# Stack into a dataset
X = np.vstack((x1, x2, x3, x4, x5)).T

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(X_scaled)

# Function to plot results
def plot_results(X_transformed, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].scatter(f1, X_transformed[:, 0], alpha=0.5)
    axes[0, 0].set_xlabel("f1")
    axes[0, 0].set_ylabel("Component 1")
    axes[0, 0].set_title("f1 vs. Component 1")
    
    axes[0, 1].scatter(f1, X_transformed[:, 1], alpha=0.5)
    axes[0, 1].set_xlabel("f1")
    axes[0, 1].set_ylabel("Component 2")
    axes[0, 1].set_title("f1 vs. Component 2")
    
    axes[1, 0].scatter(f2, X_transformed[:, 0], alpha=0.5)
    axes[1, 0].set_xlabel("f2")
    axes[1, 0].set_ylabel("Component 1")
    axes[1, 0].set_title("f2 vs. Component 1")
    
    axes[1, 1].scatter(f2, X_transformed[:, 1], alpha=0.5)
    axes[1, 1].set_xlabel("f2")
    axes[1, 1].set_ylabel("Component 2")
    axes[1, 1].set_title("f2 vs. Component 2")
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot PCA results
plot_results(X_pca, "PCA Results")

# Plot Factor Analysis results
plot_results(X_fa, "Factor Analysis Results")
