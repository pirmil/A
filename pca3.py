import numpy as np
from sklearn.decomposition import PCA

def transform_matrix_v1(matrix: np.ndarray):
    """
    For each PC, make sure that the largest loading of a feature onto this PC is positive.
    Recall that the eigenvectors are defined up to a sign.
    """
    max_indices = np.argmax(np.abs(matrix), axis=0)
    matrix[:, matrix[max_indices, np.arange(matrix.shape[1])] < 0] *= -1
    return matrix

class MyEnhancedPCA:
    def __init__(self, n_components: int):
        self.dim: int = None
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.name: str = None
        self.eivals: np.ndarray = None
        self.eivecs: np.ndarray = None

    def fit(self, X: np.ndarray):
        self.dim = X.shape[1]
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.pca.fit(self.standardize(X))
        return self

    def transform(self, Y: np.ndarray, reduce_dim: bool, rescale: bool) -> np.ndarray:
        Ypca = self.standardize(Y) @ self.pca.components_.T
        if not reduce_dim:
            Ypca = Ypca @ self.pca.components_
            if rescale:
                Ypca = self.inverse_standardize(Ypca)
        return Ypca
    
    def fit_transform(self, X, reduce_dim: bool, rescale: bool) -> np.ndarray:
        self.fit(X)
        return self.transform(X, reduce_dim, rescale)
    
    def transform_ortho(self, Y: np.ndarray, rescale: bool) -> np.ndarray:
        """
        The only way to reduce the dimension in this case is to compute 
        the components_ number n_components + 1 to self.dim. Name U these components
        The result is Ypca = self.standardize(Y) @ U
        """
        Ypca = self.standardize(Y) @ (np.eye(self.dim) - self.pca.components_.T @ self.pca.components_)
        if rescale:
            Ypca = self.inverse_standardize(Ypca)
        return Ypca
    
    def transform_ortho_reduce_dim(self, X: np.ndarray, Y: np.ndarray):
        if self.eivecs is None:
            print("Warning: The principal components will be recomputed.")
            cov_or_corr = self.compute_cov_or_corr(X)
            self.eivals, self.eivecs = np.linalg.eigh(cov_or_corr)
            largest_to_smallest = np.argsort(-self.eivals)
            self.eivals = self.eivals[largest_to_smallest]
            self.eivecs = self.eivecs[:, largest_to_smallest]
            self.eivecs = transform_matrix_v1(self.eivecs)
        Ypca = self.standardize(Y) @ self.eivecs[:, self.n_components:]
        return Ypca

    def standardize(self, Y: np.ndarray) -> np.ndarray:
        pass

    def inverse_standardize(self, Y: np.ndarray) -> np.ndarray:
        pass

    def compute_cov_or_corr(self, X: np.ndarray) -> np.ndarray:
        return np.cov(self.standardize(X), rowvar=False)

class EnhancedPCACov(MyEnhancedPCA):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.name = 'cov'

    def standardize(self, Y: np.ndarray) -> np.ndarray:
        return Y - self.mean

    def inverse_standardize(self, Y: np.ndarray) -> np.ndarray:
        return Y + self.mean
    
class EnhancedPCACorr(MyEnhancedPCA):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.name = 'corr'
        
    def standardize(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.mean) / self.std

    def inverse_standardize(self, Y: np.ndarray) -> np.ndarray:
        return Y * self.std + self.mean


def generate_observations(n_samples, n_features, mean_vector: np.ndarray, covariance_matrix: np.ndarray, seed: int):
    """
    Generate n_samples observations of n_features features with a given mean vector and covariance matrix.
    
    Parameters:
    n_samples (int): Number of observations to generate.
    n_features (int): Number of features.
    mean_vector (np.ndarray): Mean vector of shape (n_features,).
    covariance_matrix (np.ndarray): Covariance matrix of shape (n_features, n_features).
    
    Returns:
    np.ndarray: Generated observations of shape (n_samples, n_features).
    """
    np.random.seed(seed)
    if mean_vector.shape[0] != n_features:
        raise ValueError("The length of the mean vector must be equal to n_features.")
    
    if covariance_matrix.shape != (n_features, n_features):
        raise ValueError("The covariance matrix must be of shape (n_features, n_features).")
    
    observations = np.random.multivariate_normal(mean_vector, covariance_matrix, n_samples)
    return observations   

def generate_covariance_matrix(n_features):
    """
    Generate a random covariance matrix (positive definite matrix) of shape (n_features, n_features).
    
    Parameters:
    n_features (int): The number of features (dimensions) for the covariance matrix.
    
    Returns:
    np.ndarray: A positive definite covariance matrix of shape (n_features, n_features).
    """
    
    # Generate a random matrix
    A = np.random.rand(n_features, n_features)
    
    # Make the matrix symmetric
    A = (A + A.T) / 2
    
    # Ensure positive definiteness by adding n_features*I
    # This guarantees that the matrix is positive definite
    A += n_features * np.eye(n_features)
    
    return A

def reconstruction_test(X, Y: np.ndarray):
    pca_cov = EnhancedPCACov(Y.shape[1]).fit(X)
    pca_corr = EnhancedPCACorr(Y.shape[1]).fit(X)
    assert np.allclose(Y, pca_cov.transform(Y, reduce_dim=False, rescale=True))
    assert np.allclose(Y, pca_corr.transform(Y, reduce_dim=False, rescale=True))

def ortho_test(X, Y, n_components):
    pca_cov = EnhancedPCACov(n_components).fit(X)
    pca_corr = EnhancedPCACorr(n_components).fit(X)
    Ypca_ortho_red = pca_cov.transform_ortho_reduce_dim(X, Y)
    assert np.allclose(Ypca_ortho_red @ pca_cov.eivecs[:, n_components:].T, pca_cov.transform_ortho(Y, rescale=False))
    Ypca_ortho_red = pca_corr.transform_ortho_reduce_dim(X, Y)
    assert np.allclose(Ypca_ortho_red @ pca_corr.eivecs[:, n_components:].T, pca_corr.transform_ortho(Y, rescale=False))

def main(n_samples=5000, n_samples_test=500, n_features=100, n_components=10, seed=0):
    covariance_matrix = generate_covariance_matrix(n_features)
    mean_vector = np.random.randn(n_features)
    X = generate_observations(n_samples, n_features, mean_vector, covariance_matrix, seed)
    Y = generate_observations(n_samples_test, n_features, mean_vector, covariance_matrix, seed+1)

    reconstruction_test(X, Y)
    ortho_test(X, Y, n_components)
    print(f"Successfully passed all tests!")

if __name__ == '__main__':
    main()

