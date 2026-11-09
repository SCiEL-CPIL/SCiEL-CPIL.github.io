# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a rich basis function expansion.
    
    Uses a combination of polynomial, trigonometric, and Gaussian terms
    to capture a wide variety of functional forms. The parameters are
    optimized via BFGS starting from all ones.
    
    Basis functions (10 total):
    0: Constant: 1
    1: Linear: x
    2: Quadratic: x^2
    3: Cubic: x^3
    4: Sine: sin(x)
    5: Cosine: cos(x)
    6: Sine(2x): sin(2x)
    7: Cosine(2x): cos(2x)
    8: Gaussian: exp(-x^2)
    9: Gaussian(2x): exp(-2x^2)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    exp_neg_x2 = np.exp(-x2)
    exp_neg_2x2 = np.exp(-2 * x2)
    
    # Combine all basis functions with their parameters
    result = (p0 * 1.0 + 
              p1 * x + 
              p2 * x2 + 
              p3 * x3 + 
              p4 * sin_x + 
              p5 * cos_x + 
              p6 * sin_2x + 
              p7 * cos_2x + 
              p8 * exp_neg_x2 + 
              p9 * exp_neg_2x2)
    
    return result
# EVOLVE-BLOCK-END
