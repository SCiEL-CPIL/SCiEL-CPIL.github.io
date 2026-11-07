"""Symbolic regression with 1 input(s) - Chebyshev polynomial basis for optimal approximation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 Chebyshev polynomials of the first kind.
    
    Chebyshev polynomials provide near-minimax approximation on [-1, 1] and are
    well-conditioned for least-squares fitting. We map x from [-2, 2] to [-1, 1]
    and use T_0 through T_9.
    
    The 10 params are coefficients for T_0(t) through T_9(t) where t = x/2.
    T_0 = 1, T_1 = t, T_2 = 2t^2 - 1, etc.
    
    This basis is optimal for approximating smooth functions on the given domain
    and is well-conditioned, making it ideal for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = params
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1, T_1(t) = t, T_n(t) = 2t*T_{n-1}(t) - T_{n-2}(t)
    T = np.empty((10, len(t)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination: y = sum_i p_i * T_i(t)
    y = T @ p
    
    return y
# EVOLVE-BLOCK-END
