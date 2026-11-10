"""Rewritten program for symbolic regression task with a product of polynomials."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product of two polynomials.
    
    The model is P(x) * Q(x) where P and Q are polynomials of degree 4.
    P(x) = p[0] + p[1]x + p[2]x^2 + p[3]x^3 + p[4]x^4
    Q(x) = p[5] + p[6]x + p[7]x^2 + p[8]x^3 + p[9]x^4
    
    This non-linear parameterization creates a structured landscape that
    can fit functions with local amplitude modulation or sharp features
    better than a linear basis combination.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Define P(x)
    P = p[0] + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4
    
    # Define Q(x)
    Q = p[5] + p[6] * x + p[7] * x2 + p[8] * x3 + p[9] * x4
    
    # Product
    result = P * Q
    
    return result
# EVOLVE-BLOCK-END
