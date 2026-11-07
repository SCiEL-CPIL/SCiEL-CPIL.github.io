"""Symbolic regression with 2 inputs: robust Chebyshev basis with adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: kill after 140s (10s safety margin before 150s limit)
_DEADLINE = 140.0
_start_time = None


def _chebyshev_basis(x, y, max_deg):
    """
    Build a Chebyshev polynomial basis in x and y up to max_deg,
    plus bilinear cross terms. Returns list of basis functions (arrays).
    
    Basis includes:
    - T_k(x) * T_l(y) for k+l <= max_deg
    This gives a well-conditioned basis on [-1,1]^2.
    """
    n = len(x)
    basis = []
    
    for k in range(max_deg + 1):
        for l in range(max_deg + 1 - k):
            # T_k(x) * T_l(y)
            tx = _chebyshev_poly(x, k)
            ty = _chebyshev_poly(y, l)
            basis.append(tx * ty)
    
    return basis


def _chebyshev_poly(x, n):
    """Compute Chebyshev polynomial T_n(x) for array x."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return x.copy()
    else:
        T_prev = np.ones_like(x, dtype=float)
        T_curr = x.copy()
        for i in range(2, n + 1):
            T_next = 2.0 * x * T_curr - T_prev
            T_prev = T_curr
            T_curr = T_next
        return T_curr


def _build_feature_matrix(x, y, max_deg):
    """
    Build feature matrix F of shape (n, num_features) using Chebyshev basis.
    """
    basis = _chebyshev_basis(x, y, max_deg)
    # Stack into matrix: each basis function is a column
    F = np.column_stack(basis)
    return F


def _safe_mse(params, F, y_target):
    """Compute MSE with non-finite penalty."""
    global _start_time
    if _start_time is not None and time.time() - _start_time > _DEADLINE:
        return 1e30
    try:
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            return 1e30
        mse = np.mean((pred - y_target) ** 2)
        if not np.isfinite(mse):
            return 1e30
        return mse
    except Exception:
        return 1e30


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Main entry point. Uses params (length 10) as coefficients for a Chebyshev basis.
    The evaluator calls scipy.optimize.minimize with this function.
    
    We use a fixed Chebyshev basis of degree 4 (total terms = 15 for k+l<=4),
    but we only have 10 params. So we use a subset:
    - T_0(x)*T_0(y) = 1
    - T_1(x)*T_0(y) = x
    - T_0(x)*T_1(y) = y
    - T_1(x)*T_1(y) = x*y
    - T_2(x)*T_0(y) = T_2(x)
    - T_0(x)*T_2(y) = T_2(y)
    - T_2(x)*T_1(y) = T_2(x)*y
    - T_1(x)*T_2(y) = x*T_2(y)
    - T_3(x)*T_0(y) = T_3(x)
    - T_0(x)*T_3(y) = T_3(y)
    
    That's 10 terms. This captures up to degree 3 in each variable separately
    plus some cross terms.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(x)
    
    # Build the 10-term Chebyshev basis
    # T_0 = 1, T_1 = x, T_2 = 2x^2-1, T_3 = 4x^3-3x
    T0 = np.ones(n)
    T1x = x.copy()
    T1y = y.copy()
    T2x = 2.0 * x**2 - 1.0
    T2y = 2.0 * y**2 - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    T3y = 4.0 * y**3 - 3.0 * y
    
    # 10 basis functions
    f0 = T0 * T0          # constant
    f1 = T1x * T0         # x
    f2 = T0 * T1y         # y
    f3 = T1x * T1y        # x*y
    f4 = T2x * T0         # T_2(x)
    f5 = T0 * T2y         # T_2(y)
    f6 = T2x * T1y        # T_2(x)*y
    f7 = T1x * T2y        # x*T_2(y)
    f8 = T3x * T0         # T_3(x)
    f9 = T0 * T3y         # T_3(y)
    
    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    pred = F @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.full(n, np.nan)
    
    return pred
# EVOLVE-BLOCK-END
