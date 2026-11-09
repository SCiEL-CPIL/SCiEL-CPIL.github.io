"""Symbolic regression using a pure Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a linear combination of Chebyshev polynomials.
    
    The input domain [-3, 3] is mapped to [-1, 1] to utilize the standard 
    Chebyshev polynomials of the first kind, T_n(t) = cos(n * arccos(t)).
    This basis is orthogonal on [-1, 1] with respect to the weight (1-t^2)^-0.5,
    providing excellent numerical conditioning for linear regression problems.
    This allows scipy.optimize.minimize (BFGS) to reliably converge to the 
    global minimum of the least-squares objective for this linear model.
    """
    # Map x from [-3, 3] to t in [-1, 1]
    # This scaling ensures the basis functions are well-behaved and orthogonal
    # over the standard Chebyshev interval.
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) through T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, x.size), dtype=np.float64)
    
    T[0, :] = 1.0
    if x.size > 0:
        T[1, :] = t
        
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination of the basis functions
    # params is a vector of length 10
    # y = sum_{k=0}^{9} params[k] * T_k(t)
    y = T @ params
    
    return y
# EVOLVE-BLOCK-END
