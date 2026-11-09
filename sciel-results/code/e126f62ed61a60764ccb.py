"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a well-conditioned, diverse basis.
    
    Basis functions (10 parameters):
    0: Chebyshev T1(x) = x
    1: Chebyshev T2(x) = 2x^2 - 1
    2: Chebyshev T3(x) = 4x^3 - 3x
    3: sin(x)
    4: cos(x)
    5: exp(-x^2)  (Gaussian bump)
    6: exp(-0.5*x^2)  (wider Gaussian)
    7: 1/(1+x^2)  (Lorentzian)
    8: x * exp(-x^2)  (asymmetric bump)
    9: tanh(x)  (smooth step-like)
    
    This basis covers polynomial, trigonometric, exponential, and rational behaviors.
    Chebyshev polynomials provide better conditioning than raw powers.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Chebyshev polynomials (scaled to [-3,3] -> but we use standard T_n on x)
    # Actually, for x in [-3,3], standard Chebyshev T_n(x) = cos(n*arccos(x/3))*3^n? 
    # No, let's just use the standard recurrence on the raw x, or better, 
    # use the fact that we have x in [-3,3]. 
    # T_0 = 1
    # T_1 = x
    # T_2 = 2x^2 - 1
    # T_3 = 4x^3 - 3x
    
    T1 = x
    T2 = 2.0 * x * x - 1.0
    T3 = 4.0 * x * x * x - 3.0 * x
    
    # Trigonometric
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Exponential / Gaussian
    exp_neg_x2 = np.exp(-x * x)
    exp_neg_half_x2 = np.exp(-0.5 * x * x)
    
    # Rational
    lorentz = 1.0 / (1.0 + x * x)
    
    # Asymmetric
    x_exp_neg_x2 = x * exp_neg_x2
    
    # Smooth step
    tanh_x = np.tanh(x)
    
    # Combine
    result = (p[0] * T1 + 
              p[1] * T2 + 
              p[2] * T3 + 
              p[3] * sin_x + 
              p[4] * cos_x + 
              p[5] * exp_neg_x2 + 
              p[6] * exp_neg_half_x2 + 
              p[7] * lorentz + 
              p[8] * x_exp_neg_x2 + 
              p[9] * tanh_x)
    
    return result
# EVOLVE-BLOCK-END
