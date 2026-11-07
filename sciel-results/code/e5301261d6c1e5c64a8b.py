"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a degree-9 Chebyshev polynomial expansion.
    x: input array of shape (N,)
    params: array of shape (10,)
    returns: array of shape (N,)
    """
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    u = x / 2.0
    
    # Compute Chebyshev polynomials T_0(u) to T_9(u)
    # T_0(u) = 1
    # T_1(u) = u
    # T_n(u) = 2*u*T_{n-1}(u) - T_{n-2}(u)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = u
    
    for n in range(2, 10):
        T[n, :] = 2.0 * u * T[n-1, :] - T[n-2, :]
        
    # Linear combination: sum(params[i] * T[i, :])
    # params is shape (10,), T is shape (10, N)
    # We want shape (N,)
    return T @ params
# EVOLVE-BLOCK-END
