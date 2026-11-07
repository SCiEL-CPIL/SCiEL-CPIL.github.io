"""Symbolic regression with Chebyshev polynomial basis of degree 9."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on the scaled interval [-1, 1].
    
    This basis provides near-orthogonality and superior numerical conditioning
    compared to monomials or hybrid bases, which stabilizes BFGS convergence.
    It is efficient for approximating smooth functions on bounded intervals.
    
    Design:
      - T_0(t) through T_9(t) where t = (2*x - 6.1) / 5.9
    
    Total: 10 basis functions, 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to t in [-1, 1]
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9
    T = np.zeros((10, len(t)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Stack basis (N, 10)
    X = T.T
    
    return X @ p
# EVOLVE-BLOCK-END
