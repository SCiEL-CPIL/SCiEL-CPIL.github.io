"""
Symbolic regression with 2 inputs using a flexible Chebyshev tensor-product basis.

The model uses a product basis of Chebyshev polynomials in x and y, which provides
a stable, orthogonal-like representation on [-1, 1]. The 10 parameters are allocated
to:
  - x-only: T0(x), T1(x), T2(x)  (3 params)
  - y-only: T0(y), T1(y), T2(y)  (3 params)
  - interactions: T1(x)T1(y), T1(x)T2(y), T2(x)T1(y)  (3 params)
  - constant offset (1 param)

This gives a total of 10 basis functions. The Chebyshev basis is well-conditioned
on [-1, 1] and can represent smooth functions efficiently.
"""
import numpy as np
from scipy.optimize import minimize
import time


def _chebyshev_basis(t, degree):
    """Compute Chebyshev polynomials T_0(t), ..., T_degree(t) as a column vector."""
    t = np.asarray(t, dtype=np.float64).ravel()
    n = t.size
    basis = np.zeros((n, degree + 1), dtype=np.float64)
    if degree >= 0:
        basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = t
    for k in range(2, degree + 1):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev tensor-product basis.
    
    Basis (10 terms):
      p[0]  * 1
      p[1]  * T1(x)
      p[2]  * T2(x)
      p[3]  * T1(y)
      p[4]  * T2(y)
      p[5]  * T1(x)*T1(y)
      p[6]  * T1(x)*T2(y)
      p[7]  * T2(x)*T1(y)
      p[8]  * T2(x)*T2(y)
      p[9]  * x*y   (additional interaction for flexibility)
    
    Returns an array of predictions, one per data point.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if params.size != 10:
        # Pad or truncate to 10
        p = np.ones(10, dtype=np.float64)
        m = min(params.size, 10)
        p[:m] = params[:m]
        params = p
    
    # Compute Chebyshev basis up to degree 2
    # T0 = 1, T1 = t, T2 = 2t^2 - 1
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    
    # Build the 10 basis terms
    b0 = np.ones_like(x)
    b1 = T1_x
    b2 = T2_x
    b3 = T1_y
    b4 = T2_y
    b5 = T1_x * T1_y
    b6 = T1_x * T2_y
    b7 = T2_x * T1_y
    b8 = T2_x * T2_y
    b9 = x * y
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
