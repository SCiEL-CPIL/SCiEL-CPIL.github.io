"""Rewritten program for symbolic regression task with a rational function model."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rational function model:
    y(x) = P(x) / Q(x)
    
    where P(x) and Q(x) are polynomials of degree 4.
    
    10 parameters correspond to:
      p[0]..p[4]: coefficients of numerator P(x) = p[0] + p[1]x + p[2]x^2 + p[3]x^3 + p[4]x^4
      p[5]..p[9]: coefficients of denominator Q(x) = p[5] + p[6]x + p[7]x^2 + p[8]x^3 + p[9]x^4
    
    This rational form can represent functions with poles, asymptotes, and
    rapid local changes that linear combinations of global basis functions
    struggle to capture efficiently.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Numerator polynomial P(x)
    numerator = p[0] + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4
    
    # Denominator polynomial Q(x)
    denominator = p[5] + p[6] * x + p[7] * x2 + p[8] * x3 + p[9] * x4
    
    # Compute result with safe division
    # Add small epsilon to denominator to avoid division by zero
    eps = 1e-10
    result = numerator / (denominator + np.sign(denominator) * eps)
    
    # Replace any non-finite values with a large finite value
    result = np.where(np.isfinite(result), result, 1e6 * np.sign(numerator))
    
    return result
# EVOLVE-BLOCK-END
