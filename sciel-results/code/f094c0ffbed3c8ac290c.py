import numpy as np

def _legendre_basis(x):
    """
    Compute Legendre polynomials P_0 to P_9 for input x.
    x is assumed to be in [-1, 1].
    Returns an array of shape (N, 10).
    """
    N = x.shape[0]
    basis = np.empty((N, 10))
    if N == 0:
        return basis
    basis[:, 0] = 1.0
    basis[:, 1] = x
    for k in range(2, 10):
        basis[:, k] = ((2.0 * k - 1.0) * x * basis[:, k-1] - (k - 1.0) * basis[:, k-2]) / k
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1]
    t = x * 0.5
    # Compute the 10-dimensional Legendre basis
    basis = _legendre_basis(t)
    # Linear combination of basis functions
    return basis @ params
