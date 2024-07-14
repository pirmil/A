"""
This implementation of the PCA is more flexible than sklearn's implementation since it also allows to project data of shape n x d onto the PC's while keeping the original dimension d.
On the other hand, sklearn's `pca.transform` will return a n x n_components matrix, which is the same using `my_pca.transform(reduce_dim=True)`
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import time

def transform_matrix_v1(matrix: np.ndarray):
    """
    For each PC, make sure that the largest loading of a feature onto this PC is positive.
    Recall that the eigenvectors are defined up to a sign.
    """
    max_indices = np.argmax(np.abs(matrix), axis=0)
    matrix[:, matrix[max_indices, np.arange(matrix.shape[1])] < 0] *= -1
    return matrix

class MyPCA:
    def __init__(self):
        self.dim: int = None
        self.eivecs: np.ndarray = None
        self.eivals: np.ndarray = None

    def run_tests(self, X: np.ndarray, Xc: np.ndarray, cov: np.ndarray):
        n = X.shape[0]
        assert len(self.eivals) == self.dim
        assert self.eivecs.shape == (self.dim, self.dim)
        assert np.all(self.eivals[:-1] >= self.eivals[1:]), "Eigenvalues must be sorted from largest to smallest"
        assert np.allclose(cov, Xc.T @ Xc / (n - 1))
        assert np.allclose(cov, self.eivecs @ np.diag(self.eivals) @ self.eivecs.T)
        for i in range(self.dim):
            assert np.allclose(cov @ self.eivecs[:, i], self.eivals[i] * self.eivecs[:, i])
        # check that we are able to reconstruct the full input matrix
        Xt = self.transform(X, k=self.dim, reduce_dim=False, rescale=True)
        assert np.allclose(X, Xt)
        # check that projecting onto the orthogonal is the same as computing I - P_k @ P_k^T
        k = self.dim // 2 if self.dim >= 2 else 1
        X_ortho = self.transform_ortho(X, k, reduce_dim=False, rescale=True)
        assert np.allclose(X_ortho, self.inverse_standardize(self.standardize(X) @ self.eivecs[:, k:] @ self.eivecs[:, k:].T))

    def get_cumulative_explained_variance(self, pct: bool) -> np.ndarray:
        cum_var = np.cumsum(self.eivals)
        if pct:
            cum_var = cum_var / cum_var[-1]
        return cum_var
    
    def get_loadings(self, k: int, verbose=False):
        assert k <= self.dim, "The number of PCA components must be less than the data dimension"
        Lk = self.eivecs[:, :k] @ np.diag(np.sqrt(self.eivals[:k]))
        if verbose:
            print(f"Lk[i, j] is the loading of the feature i onto PC j")
            print(f"Loading of the first feature onto the PCs:")
            print(Lk[0])
        return Lk
    
    def transform(self, Y: np.ndarray, k: int, reduce_dim: bool, rescale=True) -> np.ndarray:
        """
        Transforms the input data using PCA.

        Parameters:
        Y (np.ndarray): The input data matrix of shape (n_samples, dim).
        k (int): The number of principal components to retain.
        reduce_dim (bool): If True, the output will have reduced dimensionality (n_samples, k).
                        If False, the output will be projected back to the original dimensionality (n_samples, dim).

        Returns:
        np.ndarray: The transformed data matrix. Its shape will be (n_samples, k) if reduce_dim is True,
                    otherwise it will be (n_samples, dim)
        """
        assert isinstance(Y, np.ndarray)
        assert k <= self.dim, "The number of PCA components must be less than the data dimension"
        Ypca = self.standardize(Y) @ self.eivecs[:, :k]
        if not reduce_dim:
            Ypca = Ypca @ self.eivecs[:, :k].T
            if rescale:
                Ypca = self.inverse_standardize(Ypca)
        return Ypca
    
    def transform_ortho(self, Y: np.ndarray, k: int, reduce_dim: bool, rescale=True) -> np.ndarray:
        """
        Transforms the input data using an orthogonal complement projection based on PCA.

        Parameters:
        Y (np.ndarray): The input data matrix of shape (n_samples, dim).
        k (int): The number of principal components to retain in the orthogonal complement.
        reduce_dim (bool): If True, the output will have reduced dimensionality (n_samples, k).
                        If False, the output will be projected back to the original dimensionality (n_samples, dim).

        Returns:
        np.ndarray: The transformed data matrix. Its shape will be (n_samples, k) if reduce_dim is True,
                    otherwise it will be (n_samples, dim).
        """
        assert isinstance(Y, np.ndarray)
        assert k <= self.dim, "The number of PCA components must be less than the data dimension"
        Ypca_ortho = self.standardize(Y) @ (np.eye(self.dim) - self.eivecs[:, :k] @ self.eivecs[:, :k].T)
        if reduce_dim:
            Ypca_ortho = Ypca_ortho @ self.eivecs[:, :k]
            assert np.allclose(Ypca_ortho, np.zeros_like(Ypca_ortho))
        elif rescale:
            Ypca_ortho = self.inverse_standardize(Ypca_ortho)
        return Ypca_ortho
    
    def standardize(self, Y: np.ndarray) -> np.ndarray:
        pass

    def inverse_standardize(self, Y: np.ndarray) -> np.ndarray:
        pass


class PCAWithCovariance(MyPCA):
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, verbose=True):
        assert isinstance(X, np.ndarray)
        self.dim = X.shape[1]
        self.mean = X.mean(axis=0)
        Xc = self.standardize(X)
        cov = np.cov(Xc, rowvar=False)
        eivals, eivecs = np.linalg.eigh(cov)
        largest_to_smallest = np.argsort(-eivals)
        self.eivals = eivals[largest_to_smallest]
        self.eivecs = eivecs[:, largest_to_smallest]
        self.eivecs = transform_matrix_v1(self.eivecs)
        if verbose:
            self.run_tests(X, Xc, cov)
            print(f"Successfully ran all tests!")
    
    def standardize(self, Y: np.ndarray) -> np.ndarray:
        assert Y.shape[1] == self.dim, "Input data does not have the correct dimension"
        return Y - self.mean[np.newaxis, :]
    
    def inverse_standardize(self, Y: np.ndarray) -> np.ndarray:
        assert Y.shape[1] == self.dim, "Input data does not have the correct dimension"
        return Y + self.mean[np.newaxis, :]

class PCAWithCorrelation(MyPCA):
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, verbose=True):
        assert isinstance(X, np.ndarray)
        self.dim = X.shape[1]
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        Xcr = self.standardize(X)
        cov = np.cov(Xcr, rowvar=False)
        eivals, eivecs = np.linalg.eigh(cov)
        largest_to_smallest = np.argsort(-eivals)
        self.eivals = eivals[largest_to_smallest]
        self.eivecs = eivecs[:, largest_to_smallest]
        self.eivecs = transform_matrix_v1(self.eivecs)
        if verbose:
            self.run_tests(X, Xcr, cov)
            print(f"Successfully ran all tests!")
    
    def standardize(self, Y: np.ndarray) -> np.ndarray:
        assert Y.shape[1] == self.dim, "Input data does not have the correct dimension"
        return (Y - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]
    
    def inverse_standardize(self, Y: np.ndarray) -> np.ndarray:
        assert Y.shape[1] == self.dim, "Input data does not have the correct dimension"
        return Y * self.std[np.newaxis, :] + self.mean[np.newaxis, :]
    
def test_reduce_dim_cov(X: np.ndarray, Y: np.ndarray, k: int):
    pca_cov = PCAWithCovariance()
    pca_cov.fit(X, verbose=True)
    Ycov_reduced = pca_cov.transform(Y, k, reduce_dim=True, rescale=None)

    pca_sklearn = PCA(n_components=k)
    pca_sklearn.fit(X)
    Y_sklearn = pca_sklearn.transform(Y)   

    assert np.allclose(Y_sklearn, Ycov_reduced)

def test_reduce_dim_corr(X: np.ndarray, Y: np.ndarray, k: int):
    pca_corr = PCAWithCorrelation()
    pca_corr.fit(X, verbose=True)
    Ycorr_reduced = pca_corr.transform(Y, k, reduce_dim=True, rescale=None)

    pca_sklearn = PCA(n_components=k)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_sklearn.fit(X_scaled)
    Y_scaled = scaler.transform(Y)
    Ycorr_sklearn = pca_sklearn.transform(Y_scaled)

    assert np.allclose(Ycorr_sklearn, Ycorr_reduced)

def test_project_and_rescale_cov(X: np.ndarray, Y: np.ndarray, k: int):
    pca_cov = PCAWithCovariance()
    pca_cov.fit(X, verbose=True)
    Ycov_rescaled = pca_cov.transform(Y, k, reduce_dim=False, rescale=True)

    pca_sklearn = PCA(n_components=k)
    pca_sklearn.fit(X)
    Y_sklearn = pca_sklearn.inverse_transform(pca_sklearn.transform(Y))

    assert np.allclose(Y_sklearn, Ycov_rescaled)

    Ycov_centered = pca_cov.transform(Y, k, reduce_dim=False, rescale=False)
    assert np.allclose(Y_sklearn, Ycov_centered + pca_cov.mean)

def test_project_and_rescale_corr(X: np.ndarray, Y: np.ndarray, k: int):
    pca_corr = PCAWithCorrelation()
    pca_corr.fit(X, verbose=True)
    Ycorr_rescaled = pca_corr.transform(Y, k, reduce_dim=False, rescale=True)

    pca_sklearn = PCA(n_components=k)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_sklearn.fit(X_scaled)
    Y_scaled = scaler.transform(Y)
    Ycorr_sklearn = scaler.inverse_transform(pca_sklearn.inverse_transform(pca_sklearn.transform(Y_scaled)))

    assert np.allclose(Ycorr_sklearn, Ycorr_rescaled)

    Ycorr_standardized = pca_corr.transform(Y, k, reduce_dim=False, rescale=False)
    assert np.allclose(Ycorr_sklearn, Ycorr_standardized * pca_corr.std + pca_corr.mean)    

def speed_test_1(X, Y, k):
    print("\nSpeed test 1")
    start_time = time()
    pca_cov = PCAWithCovariance()
    pca_cov.fit(X, verbose=False)
    Ycov_reduced = pca_cov.transform(Y, k, reduce_dim=True, rescale=None)
    print(f"MyPCA: {time() - start_time:.3f} seconds")

    start_time = time()
    pca_sklearn = PCA(n_components=k)
    pca_sklearn.fit(X)
    Y_sklearn = pca_sklearn.transform(Y)
    print(f"sklearn.PCA: {time() - start_time:.3f} seconds")

    assert np.allclose(Y_sklearn, Ycov_reduced)

def speed_test_2(X, Y, k):
    print("\nSpeed test 2")
    start_time = time()
    pca_corr = PCAWithCorrelation()
    pca_corr.fit(X, verbose=False)
    Ycorr_rescaled = pca_corr.transform(Y, k, reduce_dim=False, rescale=True)
    print(f"MyPCA: {time() - start_time:.3f} seconds")

    start_time = time()
    pca_sklearn = PCA(n_components=k)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_sklearn.fit(X_scaled)
    Y_scaled = scaler.transform(Y)
    Ycorr_sklearn = scaler.inverse_transform(pca_sklearn.inverse_transform(pca_sklearn.transform(Y_scaled)))
    print(f"sklearn.PCA: {time() - start_time:.3f} seconds")

    assert np.allclose(Ycorr_sklearn, Ycorr_rescaled)


def test_ortho(X: np.ndarray, Y: np.ndarray, k):
    pca_cov = PCAWithCovariance()
    pca_cov.fit(X, verbose=False)
    Ycov_rescaled = pca_cov.transform_ortho(Y, k, reduce_dim=False, rescale=True)

    pca_sklearn = PCA(n_components=k)
    pca_sklearn.fit(X)
    comp = pca_sklearn.components_
    Y_sklearn = (Y - pca_sklearn.mean_) @ (np.eye(Y.shape[1]) - comp.T @ comp) + pca_sklearn.mean_

    assert np.allclose(Ycov_rescaled, Y_sklearn)

    pca_corr = PCAWithCorrelation()
    pca_corr.fit(X, verbose=False)
    Ycorr_rescaled = pca_corr.transform_ortho(Y, k, reduce_dim=False, rescale=True)

    pca_sklearn = PCA(n_components=k)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_sklearn.fit(X_scaled)
    Y_scaled = scaler.transform(Y)
    comp = pca_sklearn.components_
    Y_sklearn = scaler.inverse_transform(Y_scaled @ (np.eye(Y.shape[1]) - comp.T @ comp))

    assert np.allclose(Ycorr_rescaled, Y_sklearn)

def main(d=5, k=3, seed=42):

    n = 300000
    n_prime = 5000
    np.random.seed(seed)
    X = np.random.randn(n, d)
    Y = np.random.randn(n_prime, d)

    test_reduce_dim_cov(X, Y, k)
    test_reduce_dim_corr(X, Y, k)
    test_project_and_rescale_cov(X, Y, k)
    test_project_and_rescale_corr(X, Y, k)
    test_ortho(X, Y, k)

    speed_test_1(X, Y, k)
    speed_test_2(X, Y, k)

    print(f"\nSuccessfully replicated sklearn PCA! The difference in speed comes from the fact that sklearn.PCA only computes n_components eigenvectors while MyPCA computes all eigenvectors.\n")
    print(f"Conclusion: Use in priority the following settings:")
    print(f"transform(Y, k, reduce_dim=True) to get the same output as sklearn.PCA().transform()")
    print(f"transform(Y, k, reduce_dim=False, rescale=True) to get the same output as sklearn.PCA().inverse_transform(transform())")
    print(f"No need to worry about using a StandardScaler because it is included in the PCAWithCorrelation class")
    print(f"Use transform_ortho(Y, k, reduce_dim=True) and transform_ortho(Y, k, reduce_dim=False, rescale=True) to project onto orthogonals!")

if __name__ == '__main__':
    main()
