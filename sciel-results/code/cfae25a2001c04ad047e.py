"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a product of 1D Fourier-like trigonometric terms.
    
    The model is a linear combination of 10 separable trigonometric product terms:
      f(x, y) = sum_{i=0}^{9} p_i * phi_i(x) * psi_i(y)
    
    Where the basis functions are chosen to capture various harmonic components:
      i=0: 1 * 1
      i=1: cos(x) * 1
      i=2: 1 * cos(y)
      i=3: sin(x) * 1
      i=4: 1 * sin(y)
      i=5: cos(2x) * 1
      i=6: 1 * cos(2y)
      i=7: sin(2x) * 1
      i=8: 1 * sin(2y)
      i=9: cos(x) * cos(y)
    
    This basis allows for independent variation in x and y up to the 2nd harmonic,
    plus a 1st-order interaction, which is more general than the previous diagonal harmonic basis.
    """
    p = params
    
    # Build the basis functions as products of 1D trigonometric terms
    # Term 0: constant
    term0 = np.ones_like(x)
    
    # Term 1: cos(x)
    term1 = np.cos(x)
    
    # Term 2: cos(y)
    term2 = np.cos(y)
    
    # Term 3: sin(x)
    term3 = np.sin(x)
    
    # Term 4: sin(y)
    term4 = np.sin(y)
    
    # Term 5: cos(2x)
    term5 = np.cos(2.0 * x)
    
    # Term 6: cos(2y)
    term6 = np.cos(2.0 * y)
    
    # Term 7: sin(2x)
    term7 = np.sin(2.0 * x)
    
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
