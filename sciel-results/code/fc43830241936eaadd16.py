"""Symbolic regression with 3 inputs using a Fourier (trigonometric) basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Fourier basis.
    
    The basis includes:
    - Constant: 1
    - Cosines of individual inputs: cos(x), cos(y), cos(z)
    - Sines of individual inputs: sin(x), sin(y), sin(z)
    - Cosines of pairwise sums: cos(x+y), cos(y+z), cos(z+x)
    
    This approach captures periodic or oscillatory components and smooth variations
    that may not be well-represented by low-degree polynomials.
    """
    p = params
    
    # Ensure finite values
    if not np.all(np.isfinite(p)):
        return np.ones_like(x)
    
    # Build the Fourier basis
    term0 = 1.0
    term1 = np.cos(x)
    term2 = np.cos(y)
    term3 = np.cos(z)
    term4 = np.sin(x)
    term5 = np.sin(y)
    term6 = np.sin(z)
    term7 = np.cos(x + y)
    term8 = np.cos(y + z)
    term9 = np.cos(z + x)
    
    result = (p[0] * term0 +
              p[1] * term1 +
              p[2] * term2 +
              p[3] * term3 +
              p[4] * term4 +
              p[5] * term5 +
              p[6] * term6 +
              p[7] * term7 +
              p[8] * term8 +
              p[9] * term9)
    
    return result
# EVOLVE-BLOCK-END
