"""Symbolic regression: nonlinear parametric model with adaptive basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear model where params control both shape and amplitude.
    
    Design: A sum of two adaptive Gaussian-like bumps in 2D plus a global trend.
    - params[0,1]: center of bump 1 (cx1, cy1)
    - params[2]:   width of bump 1
    - params[3]:   amplitude of bump 1
    - params[4,5]: center of bump 2 (cx2, cy2)
    - params[6]:   width of bump 2
    - params[7]:   amplitude of bump 2
    - params[8]:   linear coefficient for x
    - params[9]:   linear coefficient for y
    
    This gives localized nonlinearity (bumps) + global linear trend.
    """
    cx1, cy1 = params[0], params[1]
    w1 = np.maximum(params[2], 0.1)  # avoid zero/negative width
    a1 = params[3]
    
    cx2, cy2 = params[4], params[5]
    w2 = np.maximum(params[6], 0.1)
    a2 = params[7]
    
    # Gaussian bump 1
    r1_sq = (x - cx1)**2 + (y - cy1)**2
    bump1 = a1 * np.exp(-r1_sq / (2.0 * w1**2))
    
    # Gaussian bump 2
    r2_sq = (x - cx2)**2 + (y - cy2)**2
    bump2 = a2 * np.exp(-r2_sq / (2.0 * w2**2))
    
    # Linear trend
    trend = params[8] * x + params[9] * y
    
    return bump1 + bump2 + trend
# EVOLVE-BLOCK-END
