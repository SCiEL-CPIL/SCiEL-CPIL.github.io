import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid basis: Chebyshev polynomials
    (even degrees) and sine/cosine terms. This provides both polynomial and
    trigonometric flexibility within the 10-parameter budget.
    
    Basis: [1, T_2, T_4, T_6, T_8, sin(x), sin(2x), sin(3x), cos(x), cos(2x)]
    """
    n = x.shape[0]
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Constant term
    basis[:, 0] = 1.0
    
    # Even Chebyshev polynomials: T_2, T_4, T_6, T_8
    t = x / 2.0
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    for k in range(2, 10, 2):
        Tk = 2.0 * t * T1 - T0
        basis[:, k // 2] = Tk
        T0, T1 = T1, Tk
    
    # Sine terms: sin(x), sin(2x), sin(3x)
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.sin(2.0 * x)
    basis[:, 7] = np.sin(3.0 * x)
    
    # Cosine terms: cos(x), cos(2x)
    basis[:, 8] = np.cos(x)
    basis[:, 9] = np.cos(2.0 * x)
    
    return basis @ params
