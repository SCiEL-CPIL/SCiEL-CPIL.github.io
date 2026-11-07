import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev polynomial basis.
    The input x is mapped to [-1, 1] to improve numerical conditioning.
    params[0..9] are coefficients for Chebyshev polynomials T_0 to T_9.
    """
    # Map x from [0.1, 6.0] to [-1, 1]
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    basis = np.empty((len(x), 10))
    basis[:, 0] = 1.0
    if len(x) > 0:
        basis[:, 1] = t
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Compute the linear combination
    return basis @ params
