"""Symbolic regression with 3 inputs - Trigonometric Basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the hidden function using a linear combination of trigonometric and linear basis functions.
    Basis:
    1. 1
    2. x
    3. y
    4. z
    5. sin(pi * x)
    6. cos(pi * x)
    7. sin(pi * y)
    8. cos(pi * y)
    9. sin(pi * z)
    10. cos(pi * z)
    """
    p = params.flatten()
    
    # Ensure inputs are arrays for broadcasting if necessary, though they should be aligned
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Calculate basis terms
    term_0 = 1.0
    term_1 = x
    term_2 = y
    term_3 = z
    
    term_4 = np.sin(np.pi * x)
    term_5 = np.cos(np.pi * x)
    
    term_6 = np.sin(np.pi * y)
    term_7 = np.cos(np.pi * y)
    
    term_8 = np.sin(np.pi * z)
    term_9 = np.cos(np.pi * z)
    
    # Combine
    result = (p[0] * term_0 + 
              p[1] * term_1 + 
              p[2] * term_2 + 
              p[3] * term_3 + 
              p[4] * term_4 + 
              p[5] * term_5 + 
              p[6] * term_6 + 
              p[7] * term_7 + 
              p[8] * term_8 + 
              p[9] * term_9)
              
    return result
# EVOLVE-BLOCK-END
