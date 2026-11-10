"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials:
    f(x) = (a_0 T_0 + a_1 T_1 + ... + a_4 T_4) / (b_0 T_0 + b_1 T_1 + ... + b_4 T_4)
    
    This form allows for poles and better approximation of non-polynomial behaviors.
    
    params[0:5] are numerator coefficients a_0..a_4
    params[5:10] are denominator coefficients b_0..b_4
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.reshape(-1)
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(x_flat)
    T1 = x_flat.copy()
    T2 = 2.0 * x_flat * x_flat - 1.0
    T3 = 2.0 * x_flat * T2 - T1
    T4 = 2.0 * x_flat * T3 - T2
    
    # Extract coefficients
    a = params[0:5]
    b = params[5:10]
    
    # Compute numerator and denominator
    numerator = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    denominator = b[0] * T0 + b[1] * T1 + b[2] * T2 + b[3] * T3 + b[4] * T4
    
    # Handle division by zero safely
    # If denominator is very close to zero, replace it with a small value to avoid inf/nan
    # This acts as a soft constraint during optimization
    eps = 1e-6
    abs_den = np.abs(denominator)
    sign_den = np.where(denominator >= 0, 1.0, -1.0)
    safe_denominator = np.where(abs_den < eps, eps * sign_den, denominator)
    
    result = numerator / safe_denominator
    
    # Clip to prevent extreme values from affecting MSE gradients excessively
    result = np.clip(result, -1e6, 1e6)
    
    return result.reshape(original_shape)
# EVOLVE-BLOCK-END
