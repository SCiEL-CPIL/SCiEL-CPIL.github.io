"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a sum of product-of-trigonometric-terms basis.
    
    The model is:
      f(x, y) = p0 + p1*cos(a1*x)*cos(a1*y) + p2*sin(a2*x)*sin(a2*y) + ...
    
    We use 10 parameters:
      params[0] : constant offset
      params[1:3] : amplitude for cos(x)*cos(y) and sin(x)*sin(y)
      params[3:5] : amplitude for cos(2x)*cos(2y) and sin(2x)*sin(2y)
      params[5:7] : amplitude for cos(3x)*cos(3y) and sin(3x)*sin(3y)
      params[7:9] : amplitude for cos(4x)*cos(4y) and sin(4x)*sin(4y)
      params[9]  : amplitude for sin(5x)*sin(5y)
    
    This basis captures periodic/oscillatory structures in both x and y directions.
    """
    p = params
    
    # Build the basis functions
    # Term 0: constant
    term0 = np.ones_like(x)
    
    # Term 1: cos(x)*cos(y)
    term1 = np.cos(x) * np.cos(y)
    
    # Term 2: sin(x)*sin(y)
    term2 = np.sin(x) * np.sin(y)
    
    # Term 3: cos(2x)*cos(2y)
    term3 = np.cos(2.0 * x) * np.cos(2.0 * y)
    
    # Term 4: sin(2x)*sin(2y)
    term4 = np.sin(2.0 * x) * np.sin(2.0 * y)
    
    # Term 5: cos(3x)*cos(3y)
    term5 = np.cos(3.0 * x) * np.cos(3.0 * y)
    
    # Term 6: sin(3x)*sin(3y)
    term6 = np.sin(3.0 * x) * np.sin(3.0 * y)
    
    # Term 7: cos(4x)*cos(4y)
    term7 = np.cos(4.0 * x) * np.cos(4.0 * y)
    
    # Term 8: sin(4x)*sin(4y)
    term8 = np.sin(4.0 * x) * np.sin(4.0 * y)
    
    # Term 9: sin(5x)*sin(5y)
    term9 = np.sin(5.0 * x) * np.sin(5.0 * y)
    
    # Combine with parameters
    prediction = (
        p[0] * term0 +
        p[1] * term1 +
        p[2] * term2 +
        p[3] * term3 +
        p[4] * term4 +
        p[5] * term5 +
        p[6] * term6 +
        p[7] * term7 +
        p[8] * term8 +
        p[9] * term9
    )
    
    return prediction
# EVOLVE-BLOCK-END
