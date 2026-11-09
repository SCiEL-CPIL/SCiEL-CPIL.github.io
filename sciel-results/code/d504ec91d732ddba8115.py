"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a bounded Chebyshev polynomial basis up to degree 9.
    This provides a well-conditioned representation for smooth functions
    on [-3, 3], allowing BFGS to converge to a better fit than raw polynomials
    or mixed bases.
    """
    # Map x from [-3, 3] to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
    
    T = np.zeros((10, len(t)))
    T[0] = 1.0
    T[1] = t
    
    for n in range(2, 10):
        T[n] = 2.0 * t * T[n-1] - T[n-2]
    
    # Combine with parameters
    return np.sum(params[:, np.newaxis] * T, axis=0)
# EVOLVE-BLOCK-END
