"""Symmetric regression with orthogonal Chebyshev basis for stable fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_basis(x: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute Chebyshev polynomials of the first kind T_k(x) for k=0..n-1.
    Maps x from [-2, 2] to [-1, 1] for standard Chebyshev domain.
    Returns shape (len(x), n).
    """
    x = np.asarray(x, dtype=float)
    # Map from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # T_0(t) = 1, T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    basis = np.zeros((len(x), n), dtype=float)
    if n > 0:
        basis[:, 0] = 1.0
    if n > 1:
        basis[:, 1] = t
    for k in range(2, n):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using orthogonal Chebyshev basis.
    params: array of 10 coefficients.
    Returns predictions for each point in x.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Compute Chebyshev basis matrix: (n_points, 10)
    basis = _chebyshev_basis(x, n=10)
    
    # Prediction is basis @ params
    prediction = basis @ params
    
    # Ensure output is 1D array with one prediction per point
    return prediction
# EVOLVE-BLOCK-END
