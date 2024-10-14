# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Loading the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardizing the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initializing PCA (2 components for visualization)
pca = PCA(n_components=2)

# Applying PCA on the dataset
X_pca = pca.fit_transform(X_scaled)

# Displaying the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio by each principal component: {explained_variance}")

# Converting PCA results to a DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y

# Print first few rows of PCA results
print(pca_df.head())
