import numpy as np
from sklearn.decomposition import PCA

def is_orthonormal_basis(matrix: np.ndarray):
    """
    Check if a given matrix spans an orthonormal basis.

    Parameters:
    matrix (numpy.ndarray): A matrix of shape (k, d) containing k vectors of dimension d.

    Returns:
    bool: True if the matrix spans an orthonormal basis, False otherwise.

    The function checks two conditions:
    1. Orthonormality: The dot product of the matrix with its transpose should be the identity matrix of size k.
    2. Rank: The matrix should have full rank, i.e., rank k.
    """
    k, d = matrix.shape
    identity_approx = np.dot(matrix, matrix.T)
    identity = np.eye(k)
    orthonormal = np.allclose(identity_approx, identity, atol=1e-8)
    rank = np.linalg.matrix_rank(matrix)
    return orthonormal and rank == k

def project_data_v2(data: np.ndarray, basis: np.ndarray):
    """
    Project data points onto a subspace defined by an orthonormal basis.
    
    Parameters:
    data (numpy.ndarray): A matrix of shape (N, d) where each row is a data point in d-dimensional space.
    basis (numpy.ndarray): A matrix of shape (k, d) containing k orthonormal vectors (each row is an orthonormal vector).
    
    Returns:
    numpy.ndarray: A matrix of shape (N, d) where each row is the projection of the corresponding data point onto the subspace spanned by the basis vectors.
    """
    assert is_orthonormal_basis(basis)
    coefficients = data @ basis.T
    projected_data = coefficients @ basis
    return projected_data

def project_data_with_loop(data: np.ndarray, basis: np.ndarray):
    """
    Project data points onto a subspace defined by an orthonormal basis using a for loop.
    
    Parameters:
    data (numpy.ndarray): A matrix of shape (N, d) where each row is a data point in d-dimensional space.
    basis (numpy.ndarray): A matrix of shape (k, d) containing k orthonormal vectors (each row is an orthonormal vector).
    
    Returns:
    numpy.ndarray: A matrix of shape (N, d) where each row is the projection of the corresponding data point onto the subspace spanned by the basis vectors.
    """
    assert is_orthonormal_basis(basis)
    N, d = data.shape
    k, d = basis.shape
    data_proj = np.zeros((N, d))
    
    for i in range(N):
        projection = np.zeros(d)
        for j in range(k):
            coefficient = np.dot(data[i], basis[j])
            projection += coefficient * basis[j]
        data_proj[i] = projection
    return data_proj

def project_onto_first_PCs(data: np.ndarray, k: int):
    """
    Project data points onto the first k principal components.

    Parameters:
    data (numpy.ndarray): A matrix of shape (N, d) where each row is a data point in d-dimensional space.
    k (int): The number of principal components to project onto. Must be less than d.

    Returns:
    numpy.ndarray: A matrix of shape (N, d) where each row is the projection of the corresponding data point 
                   onto the subspace spanned by the first k principal components.

    Raises:
    AssertionError: If k is not less than the number of dimensions d.
    """
    assert k < data.shape[1]
    pca = PCA(n_components=k)
    pca.fit(data)
    components = pca.components_
    data_proj = project_data_v2(data, components)
    return data_proj