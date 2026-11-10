"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a 9th-degree Chebyshev series.
    The input x is scaled to [-1, 1] to match the Chebyshev domain.
    Chebyshev polynomials provide a better-conditioned basis than monomials,
    improving the stability of the BFGS optimization.
    params: array of length 10 [c0, c1, ..., c9]
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Chebyshev polynomials of the first kind T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T_prev = np.ones_like(x, dtype=float)  # T_0
    T_curr = t.copy()                       # T_1
    
    y = np.zeros_like(x, dtype=float)
    y += params[0] * T_prev
    y += params[1] * T_curr
    
    for n in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        y += params[n] * T_next
        T_prev = T_curr
        T_curr = T_next
        
    return y
# EVOLVE-BLOCK-END
