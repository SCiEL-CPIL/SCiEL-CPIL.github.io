import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 learnable parameters.
    Uses a basis of linear, interaction, triple product, and exponential terms.
    """
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    
    # Pairwise interactions
    b4 = x * y
    b5 = x * z
    b6 = y * z
    
    # Triple interaction
    b7 = x * y * z
    
    # Exponentials
    b8 = np.exp(x)
    b9 = np.exp(y)
    
    # Combine with learnable parameters
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
            params[4] * b4 + params[5] * b5 + params[6] * b6 +
            params[7] * b7 + params[8] * b8 + params[9] * b9)
