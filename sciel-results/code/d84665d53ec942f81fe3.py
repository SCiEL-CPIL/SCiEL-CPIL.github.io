import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation (Padé-like form) for a nonlinear oscillator.
    
    Structure:
    y = N(x, v) / D(x, v)
    
    Where N is a polynomial in x and v up to degree 4, and D is 1 + polynomial in x and v up to degree 4.
    This structure provides inherent saturation and better conditioning than pure polynomials,
    potentially capturing smooth, bounded nonlinear dynamics more effectively.
    
    Parameters (10 total):
    p[0..3]: Numerator coefficients for terms: 1, x, v, x^2
    p[4..7]: Denominator coefficients for terms: x, v, x^2, v^2
    p[8..9]: Additional numerator terms: x^3, x*v
    """
    # Normalization constants based on training ranges
    # x: [-0.5728, 0.5275] -> mean ~ -0.02265, scale ~ 0.55015 (half-range)
    # v: [-0.1754, 0.2821] -> mean ~ 0.05335, scale ~ 0.22875 (half-range)
    
    x_mean = -0.02265
    x_scale = 0.55015
    v_mean = 0.05335
    v_scale = 0.22875
    
    # Normalize inputs to approximately [-1, 1]
    x_n = (x - x_mean) / x_scale
    v_n = (v - v_mean) / v_scale
    
    p = params
    
    # Numerator terms
    n1 = np.ones_like(x_n)
    n2 = x_n
    n3 = v_n
    n4 = x_n * x_n
    n5 = x_n * x_n * x_n
    n6 = x_n * v_n
    
    numerator = (p[0] * n1 + 
                 p[1] * n2 + 
                 p[2] * n3 + 
                 p[3] * n4 + 
                 p[8] * n5 + 
                 p[9] * n6)
    
    # Denominator terms (1 + ...)
    d1 = np.ones_like(x_n)
    d2 = x_n
    d3 = v_n
    d4 = x_n * x_n
    d5 = v_n * v_n
    
    denominator = (d1 + 
                   p[4] * d2 + 
                   p[5] * d3 + 
                   p[6] * d4 + 
                   p[7] * d5)
    
    # Avoid division by zero
    # Use a small epsilon to prevent singularity
    eps = 1e-8
    denominator_safe = np.where(np.abs(denominator) < eps, eps, denominator)
    
    return numerator / denominator_safe
