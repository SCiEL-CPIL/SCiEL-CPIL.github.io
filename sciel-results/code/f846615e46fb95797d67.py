"""Symbolic regression with Chebyshev basis for 2D inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis_1d(t, max_degree):
    """Compute Chebyshev polynomials T_0(t), T_1(t), ..., T_max_degree(t).
    Returns a 2D array of shape (n, max_degree+1).
    """
    n = len(t)
    basis = np.zeros((n, max_degree + 1))
    basis[:, 0] = 1.0
    if max_degree >= 1:
        basis[:, 1] = t
    for k in range(2, max_degree + 1):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    n = len(x)
    
    # Compute Chebyshev basis for x and y up to degree 3
    # T_0, T_1, T_2, T_3 for each
    max_deg = 3
    bx = _chebyshev_basis_1d(x, max_deg)  # shape (n, 4)
    by = _chebyshev_basis_1d(y, max_deg)  # shape (n, 4)
    
    # Build a set of 10 features from Chebyshev polynomials:
    # Features chosen to capture common functional forms:
    # 0: T_0(x)*T_0(y) = 1 (constant)
    # 1: T_1(x) = x
    # 2: T_1(y) = y
    # 3: T_1(x)*T_1(y) = x*y
    # 4: T_2(x) = 2x^2 - 1
    # 5: T_2(y) = 2y^2 - 1
    # 6: T_2(x)*T_0(y) = 2x^2 - 1
    # 7: T_0(x)*T_2(y) = 2y^2 - 1
    # 8: T_3(x) = 4x^3 - 3x
    # 9: T_3(y) = 4y^3 - 3y
    
    # Actually, let me be more systematic. I'll use 10 features:
    # f0 = 1
    # f1 = T_1(x)
    # f2 = T_1(y)
    # f3 = T_1(x)*T_1(y)
    # f4 = T_2(x)
    # f5 = T_2(y)
    # f6 = T_2(x)*T_1(y)
    # f7 = T_1(x)*T_2(y)
    # f8 = T_3(x)
    # f9 = T_3(y)
    
    f0 = np.ones(n)
    f1 = bx[:, 1]
    f2 = by[:, 1]
    f3 = bx[:, 1] * by[:, 1]
    f4 = bx[:, 2]
    f5 = by[:, 2]
    f6 = bx[:, 2] * by[:, 1]
    f7 = bx[:, 1] * by[:, 2]
    f8 = bx[:, 3]
    f9 = by[:, 3]
    
    # Stack into design matrix (n, 10)
    A = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # Prediction: A @ params
    pred = A @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback: return mean of y (or zeros) if anything goes wrong
        return np.zeros(n)
    
    return pred
# EVOLVE-BLOCK-END
