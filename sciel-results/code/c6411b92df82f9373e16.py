"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes predictions using a complete Chebyshev polynomial basis of total degree <= 3.
    The basis is well-conditioned on [-1, 1], facilitating BFGS convergence for the
    linear least-squares problem.
    """
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2 * x**2 - 1
    T3 = 4 * x**3 - 3 * x
    
    U0 = np.ones_like(y)
    U1 = y
    U2 = 2 * y**2 - 1
    U3 = 4 * y**3 - 3 * y

    # Construct the design matrix columns (10 parameters)
    # Order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    
    # We can compute the result directly to avoid storing the full matrix if memory were a concern,
    # but for 200 points, a 200x10 matrix is trivial.
    # However, direct computation is faster and cleaner.
    
    p = params
    
    pred = (
        p[0] * T0 * U0 +   # T0(x)T0(y)
        p[1] * T1 * U0 +   # T1(x)T0(y)
        p[2] * T0 * U1 +   # T0(x)T1(y)
        p[3] * T2 * U0 +   # T2(x)T0(y)
        p[4] * T1 * U1 +   # T1(x)T1(y)
        p[5] * T0 * U2 +   # T0(x)T2(y)
        p[6] * T3 * U0 +   # T3(x)T0(y)
        p[7] * T2 * U1 +   # T2(x)T1(y)
        p[8] * T1 * U2 +   # T1(x)T2(y)
        p[9] * T0 * U3     # T0(x)T3(y)
    )
    
    return pred
# EVOLVE-BLOCK-END
