"""Symbolic regression with 2 inputs: robust Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _chebyshev_poly(x, n):
    """Compute Chebyshev polynomial T_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        t_nm1 = np.ones_like(x)
        t_n = x.copy()
        for _ in range(2, n + 1):
            t_np1 = 2.0 * x * t_n - t_nm1
            t_nm1 = t_n
            t_n = t_np1
        return t_n

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev basis model with 10 parameters.
    f(x, y) = sum_{i=0}^{9} p_i * T_{k_i}(x) * T_{l_i}(y)
    
    Basis functions (T_k(x) * T_l(y)):
    0: T_0(x) * T_0(y) = 1
    1: T_1(x) * T_0(y) = x
    2: T_0(x) * T_1(y) = y
    3: T_2(x) * T_0(y) = 2x^2 - 1
    4: T_0(x) * T_2(y) = 2y^2 - 1
    5: T_1(x) * T_1(y) = xy
    6: T_3(x) * T_0(y) = 4x^3 - 3x
    7: T_0(x) * T_3(y) = 4y^3 - 3y
    8: T_2(x) * T_1(y) = (2x^2 - 1)y
    9: T_1(x) * T_2(y) = x(2y^2 - 1)
    
    This is a linear model in the parameters, which BFGS can optimize efficiently.
    Chebyshev polynomials provide a well-conditioned basis on [-1, 1].
    """
    p = params
    
    # Pre-compute Chebyshev polynomials up to degree 3
    x0 = np.ones_like(x)
    x1 = x
    x2 = 2.0 * x * x - 1.0
    x3 = 4.0 * x * x * x - 3.0 * x
    
    y0 = np.ones_like(y)
    y1 = y
    y2 = 2.0 * y * y - 1.0
    y3 = 4.0 * y * y * y - 3.0 * y
    
    # Build the 10 basis functions
    b0 = x0 * y0  # 1
    b1 = x1 * y0  # x
    b2 = x0 * y1  # y
    b3 = x2 * y0  # 2x^2 - 1
    b4 = x0 * y2  # 2y^2 - 1
    b5 = x1 * y1  # xy
    b6 = x3 * y0  # 4x^3 - 3x
    b7 = x0 * y3  # 4y^3 - 3y
    b8 = x2 * y1  # (2x^2 - 1)y
    b9 = x1 * y2  # x(2y^2 - 1)
    
    # Linear combination
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
