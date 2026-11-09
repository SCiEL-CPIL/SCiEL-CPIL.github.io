"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_polynomials(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute Chebyshev polynomials T_0(x) through T_{n-1}(x).
    Returns an array of shape (..., n) where the last axis indexes the polynomial degree.
    Uses the three-term recurrence: T_0 = 1, T_1 = x, T_{k} = 2*x*T_{k-1} - T_{k-2}.
    """
    shape = x.shape
    x_flat = x.reshape(-1)
    n_points = x_flat.size
    
    T = np.zeros((n_points, n))
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = x_flat
    for k in range(2, n):
        T[:, k] = 2.0 * x_flat * T[:, k-1] - T[:, k-2]
    
    return T.reshape(shape + (n,))


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials:
    f(x) = (a_0 T_0 + a_1 T_1 + ... + a_4 T_4) / (b_0 T_0 + b_1 T_1 + ... + b_4 T_4)
    
    params[0:5] are numerator coefficients a_0..a_4
    params[5:10] are denominator coefficients b_0..b_4
    """
    # Ensure x is a 1D array for consistent handling
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.reshape(-1)
    
    # Compute Chebyshev polynomials T_0 through T_4
    # T_0 = 1, T_1 = x, T_2 = 2x^2 - 1, T_3 = 4x^3 - 3x, T_4 = 8x^4 - 8x^2 + 1
    T0 = np.ones_like(x_flat)
    T1 = x_flat.copy()
    T2 = 2.0 * x_flat * x_flat - 1.0
    T3 = 2.0 * x_flat * T2 - T1
    T4 = 2.0 * x_flat * T3 - T2
    
    # Build numerator and denominator
    a = params[0:5]
    b = params[5:10]
    
    numerator = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    denominator = b[0] * T0 + b[1] * T1 + b[2] * T2 + b[3] * T3 + b[4] * T4
    
    # Avoid division by zero: use a small epsilon where denominator is near zero
    eps = 1e-10
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + 1e-30), denominator)
    # Ensure sign is preserved and no zeros
    safe_denominator = np.where(safe_denominator == 0, eps, safe_denominator)
    
    result = numerator / safe_denominator
    
    # Clip to avoid extreme values that could cause numerical issues
    result = np.clip(result, -1e6, 1e6)
    
    return result.reshape(original_shape)
# EVOLVE-BLOCK-END
