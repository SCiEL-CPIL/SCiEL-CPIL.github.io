"""Symbolic regression with a hybrid Chebyshev + trigonometric + Gaussian basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model combining:
    - 5 Chebyshev polynomials (T_0 to T_4) on scaled x in [-1, 1]
    - sin(x), cos(x)
    - Two Gaussian bumps: exp(-(x-c1)^2/s^2), exp(-(x-c2)^2/s^2)
    
    params (length 10):
      0-4: Chebyshev coefficients a0..a4
      5: sin(x) coefficient
      6: cos(x) coefficient
      7: Gaussian 1 center
      8: Gaussian 2 center
      9: Gaussian amplitude (shared, with fixed width)
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Scale x to [-1, 1] for Chebyshev stability
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0(t) ... T_4(t)
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Trigonometric terms
    sinx = np.sin(x)
    cosx = np.cos(x)
    
    # Gaussian bumps with fixed width for stability
    # Use centers from params[7] and params[8], amplitude from params[9]
    # Width fixed at 1.0 for simplicity and numerical stability
    width = 1.0
    c1 = p[7]
    c2 = p[8]
    amp = p[9]
    
    # Clip centers to [-2, 2] to keep Gaussians relevant on the domain
    c1_clipped = np.clip(c1, -2.0, 2.0)
    c2_clipped = np.clip(c2, -2.0, 2.0)
    
    gauss1 = amp * np.exp(-((x - c1_clipped) ** 2) / (width * width))
    gauss2 = amp * np.exp(-((x - c2_clipped) ** 2) / (width * width))
    
    # Combine all terms
    result = (p[0] * T0 + 
              p[1] * T1 + 
              p[2] * T2 + 
              p[3] * T3 + 
              p[4] * T4 + 
              p[5] * sinx + 
              p[6] * cosx + 
              gauss1 + 
              gauss2)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
