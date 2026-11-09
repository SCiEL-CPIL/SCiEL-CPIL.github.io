"""Rewritten program for symbolic regression with Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear model using Chebyshev polynomials of the first kind.
    
    The input x is in [-2, 2]. We map to z = x/2 in [-1, 1] for stable
    Chebyshev evaluation. The model is:
        f(x) = sum_{k=0}^{9} params[k] * T_k(x/2)
    
    This is a linear model in params, so BFGS (or any convex optimizer)
    converges to the global minimum of the MSE.
    """
    z = x / 2.0
    
    # Chebyshev polynomials T_0 through T_9
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * T1 - T0
    T3 = 2.0 * z * T2 - T1
    T4 = 2.0 * z * T3 - T2
    T5 = 2.0 * z * T4 - T3
    T6 = 2.0 * z * T5 - T4
    T7 = 2.0 * z * T6 - T5
    T8 = 2.0 * z * T7 - T6
    T9 = 2.0 * z * T8 - T7
    
    # Combine with parameters
    return (params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3 +
            params[4] * T4 + params[5] * T5 + params[6] * T6 + params[7] * T7 +
            params[8] * T8 + params[9] * T9)
# EVOLVE-BLOCK-END
