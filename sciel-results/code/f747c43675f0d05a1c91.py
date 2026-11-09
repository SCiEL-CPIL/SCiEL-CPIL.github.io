"""Symbolic regression with 1 input(s) - Chebyshev polynomial approach."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a Chebyshev polynomial series of degree 9 to recover the target function.
    
    The input x is in [-2, 2]. We scale to t in [-1, 1] for numerical stability
    of the Chebyshev basis. params contains 10 coefficients for T_0(t) through T_9(t).
    
    The evaluator uses scipy.optimize.minimize(method='BFGS') to fit params,
    so we just need to provide a smooth, well-conditioned model.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) through T_9(t) iteratively
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # We'll compute the result as a linear combination of the basis functions
    # This is efficient and avoids building a large design matrix
    
    n_points = len(t)
    result = np.zeros(n_points, dtype=np.float64)
    
    if len(params) < 1:
        return result
    
    # T_0
    T_prev2 = np.ones(n_points, dtype=np.float64)  # T_0
    result += params[0] * T_prev2
    
    if len(params) < 2:
        return result
    
    # T_1
    T_prev1 = t.copy()  # T_1
    result += params[1] * T_prev1
    
    # T_n for n = 2 to 9
    for n in range(2, 10):
        if n >= len(params):
            break
        T_curr = 2.0 * t * T_prev1 - T_prev2
        result += params[n] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
    
    return result
# EVOLVE-BLOCK-END
