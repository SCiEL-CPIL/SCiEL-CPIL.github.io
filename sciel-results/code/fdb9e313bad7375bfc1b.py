"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich, well-conditioned basis of 10 terms.
    
    The basis is designed to be well-conditioned and cover a wide range of
    possible target functions:
    - Chebyshev polynomials T_0 through T_4 (5 terms) for polynomial-like behavior
    - sin(x), cos(x) for periodic behavior
    - exp(-x^2) for a Gaussian bump (localized non-polynomial)
    - tanh(x/3) for saturation behavior
    - A narrow Gaussian bump at 0 for sharp localized features
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_4(t)
    T0 = np.ones_like(t_safe)
    T1 = t_safe.copy()
    T2 = 2.0 * t_safe * T1 - T0
    T3 = 2.0 * t_safe * T2 - T1
    T4 = 2.0 * t_safe * T3 - T2
    
    # Trigonometric terms (using x directly, since domain is [-3,3])
    sin1 = np.sin(x)
    cos1 = np.cos(x)
    
    # Gaussian bump centered at 0 (width ~1 in t-space)
    gauss_wide = np.exp(-t_safe**2)
    
    # Narrow Gaussian bump centered at 0 (width ~0.5 in t-space)
    gauss_narrow = np.exp(-4.0 * t_safe**2)
    
    # Saturation
    tanh_term = np.tanh(t_safe)
    
    # Linear combination of 10 terms
    # T0, T1, T2, T3, T4 (5) + sin1, cos1 (2) + gauss_wide, gauss_narrow (2) + tanh (1) = 10
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * sin1 +
              params[6] * cos1 +
              params[7] * gauss_wide +
              params[8] * gauss_narrow +
              params[9] * tanh_term)
    
    # Ensure the result is finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
