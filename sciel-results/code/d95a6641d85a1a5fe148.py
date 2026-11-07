"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Chebyshev series of degree 9 on scaled x in [-1,1].
    This is a linear-in-parameters model, so BFGS will find the exact
    least-squares solution (up to numerical precision) in one step.
    
    The Chebyshev basis provides near-optimal polynomial approximation
    with bounded error, making it ideal for smooth target functions.
    """
    # Map x from [-2, 2] to [-1, 1]
    x_scaled = np.clip(x / 2.0, -1.0, 1.0)
    
    # Precompute Chebyshev polynomials T_0 through T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    T = np.zeros((10, len(x_scaled)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = x_scaled
    for n in range(2, 10):
        T[n, :] = 2.0 * x_scaled * T[n-1, :] - T[n-2, :]
    
    # result = sum_{n=0}^{9} params[n] * T[n](x_scaled)
    result = params @ T
    return result
# EVOLVE-BLOCK-END
