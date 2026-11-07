"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a flexible low-order trigonometric basis.
    
    The model is a linear combination of separable trigonometric terms and one interaction term:
      f(x, y) = p0 + p1*cos(x) + p2*sin(x) + p3*cos(y) + p4*sin(y) +
                p5*cos(2x) + p6*sin(2x) + p7*cos(2y) + p8*sin(2y) + p9*cos(x)*cos(y)
    
    This basis allows for independent variation in x and y up to the 2nd harmonic,
    plus a 1st-order interaction, which is more general than the previous diagonal harmonic basis.
    """
    p = params
    
    # Build the basis functions
    # Term 0: constant
    term0 = np.ones_like(x)
    
    # Term 1: cos(x)
    term1 = np.cos(x)
    
    # Term 2: sin(x)
    term2 = np.sin(x)
    
    # Term 3: cos(y)
    term3 = np.cos(y)
    
    # Term 4: sin(y)
    term4 = np.sin(y)
    
    # Term 5: cos(2x)
    term5 = np.cos(2.0 * x)
    
    # Term 6: sin(2x)
    term6 = np.sin(2.0 * x)
    
    # Term 7: cos(2y)
    term7 = np.cos(2.0 * y)
    
    # Term 8: sin(2y)
    term8 = np.sin(2.0 * y)
    
    # Term 9: cos(x)*cos(y)
    term9 = np.cos(x) * np.cos(y)
    
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
