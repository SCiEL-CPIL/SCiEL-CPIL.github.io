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
        # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    n = len(x)
    
    # Compute Chebyshev basis for x and y up to degree 3
    max_deg = 3
    bx = _chebyshev_basis_1d(x, max_deg)  # shape (n, 4)
    by = _chebyshev_basis_1d(y, max_deg)  # shape (n, 4)
    
    # Build 10 features: Total degree <= 3
    # Indices (i, j) such that i + j <= 3
    # (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    
    f0 = bx[:, 0] * by[:, 0]  # 1
    f1 = bx[:, 1] * by[:, 0]  # x
    f2 = bx[:, 0] * by[:, 1]  # y
    f3 = bx[:, 2] * by[:, 0]  # T2(x)
    f4 = bx[:, 1] * by[:, 1]  # x*y
    f5 = bx[:, 0] * by[:, 2]  # T2(y)
    f6 = bx[:, 3] * by[:, 0]  # T3(x)
    f7 = bx[:, 2] * by[:, 1]  # T2(x)*y
    f8 = bx[:, 1] * by[:, 2]  # x*T2(y)
    f9 = bx[:, 0] * by[:, 3]  # T3(y)
    
    A = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    pred = A @ params
    
    if not np.all(np.isfinite(pred)):
        return np.zeros(n)
    
    # Safety margin check (though this function is very fast)
    if time.time() - start_time > 145:
         return np.zeros(n)
         
    return pred
# EVOLVE-BLOCK-END
