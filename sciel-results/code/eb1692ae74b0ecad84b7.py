"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a 9th-order Chebyshev series on scaled domain.
    x is scaled from [0.1, 6] to t in [-1, 1].
    Basis: T_0(t), T_1(t), ..., T_9(t)
    """
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to t in [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    t = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 through T_9
    T = np.empty((10, len(t)), dtype=float)
    T[0] = 1.0
    T[1] = t
    for n in range(2, 10):
        T[n] = 2.0 * t * T[n-1] - T[n-2]
    
    # Combine with params
    return np.sum(params[:, np.newaxis] * T, axis=0)
# EVOLVE-BLOCK-END
