import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@pytest.fixture
def dataset():
    """Fixture for loading the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

@pytest.fixture
def scaled_data(dataset):
    """Fixture for scaling the dataset."""
    X, y = dataset
    scaler = StandardScaler()
    return scaler.fit_transform(X), y

def test_pca_variance(scaled_data):
    """Test that PCA preserves a significant amount of variance."""
    X_scaled, y = scaled_data
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    assert sum(explained_variance) > 0.9, f"Explained variance is too low: {sum(explained_variance)}"

def test_pca_transformation_shape(scaled_data):
    """Test that PCA transformation returns the correct shape."""
    X_scaled, y = scaled_data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    assert X_pca.shape[1] == 2, "PCA did not reduce the data to 2 dimensions"
