"""
Strategy: Bilinear Chebyshev Basis for Symbolic Regression.

Uses products of low-degree Chebyshev polynomials T_i(x) * T_j(x) to form
a bilinear basis that captures sharp non-polynomial features (like |x| or
cubic transitions) more efficiently than a single linear basis.

The 10 basis functions are:
- T_0(x) * T_0(x) = 1
- T_1(x) * T_0(x) = x
- T_2(x) * T_0(x) = 2x^2 - 1
- T_1(x) * T_1(x) = x^2
- T_3(x) * T_0(x) = 4x^3 - 3x
- T_2(x) * T_1(x) = 2x^3 - x
- T_4(x) * T_0(x) = 8x^4 - 8x^2 + 1
- T_3(x) * T_1(x) = 4x^4 - 3x^2
- T_2(x) * T_2(x) = (2x^2-1)^2 = 4x^4 - 4x^2 + 1
- T_4(x) * T_1(x) = 8x^5 - 8x^3 + x

This representation is linear in the 10 coefficients, ensuring BFGS converges
to the global minimum while exploiting multiplicative interactions.
"""
import numpy as np
from scipy.special import eval_chebyt

def _build_basis(x: np.ndarray) -> np.ndarray:
    """Build the 10-column bilinear Chebyshev basis matrix for input x."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    
    # Compute Chebyshev polynomials up to degree 4
    T0 = eval_chebyt(0, x)  # 1
    T1 = eval_chebyt(1, x)  # x
    T2 = eval_chebyt(2, x)  # 2x^2 - 1
    T3 = eval_chebyt(3, x)  # 4x^3 - 3x
    T4 = eval_chebyt(4, x)  # 8x^4 - 8x^2 + 1
    
    # Build bilinear basis: T_i(x) * T_j(x) for selected pairs
    phi = np.empty((n, 10), dtype=float)
    phi[:, 0] = T0 * T0  # 1
    phi[:, 1] = T1 * T0  # x
    phi[:, 2] = T2 * T0  # 2x^2 - 1
    phi[:, 3] = T1 * T1  # x^2
    phi[:, 4] = T3 * T0  # 4x^3 - 3x
    phi[:, 5] = T2 * T1  # 2x^3 - x
    phi[:, 6] = T4 * T0  # 8x^4 - 8x^2 + 1
    phi[:, 7] = T3 * T1  # 4x^4 - 3x^2
    phi[:, 8] = T2 * T2  # (2x^2-1)^2
    phi[:, 9] = T4 * T1  # 8x^5 - 8x^3 + x
    
    return phi

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    params = np.asarray(params, dtype=float).ravel()
    
    # Ensure params is length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    phi = _build_basis(x)
    result = phi @ params
    result = np.where(np.isfinite(result), result, 0.0)
    return result
