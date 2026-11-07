"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with interaction terms to capture oscillatory
    and diagonal structures in the unknown function.
    
    Uses sin/cos of linear combinations (x+y, x-y, x, y) to form a
    richer basis that can represent periodic and diagonal patterns.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Interaction variables
    u = x + y   # diagonal
    v = x - y   # anti-diagonal
    
    # Fourier basis: constant + 4 frequency-1 terms + 4 frequency-2 terms
    # This gives us 9 terms; we add one more for a constant offset
    b0 = np.ones_like(x)
    b1 = np.sin(u)
    b2 = np.cos(u)
    b3 = np.sin(v)
    b4 = np.cos(v)
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(y)
    b8 = np.cos(y)
    b9 = np.sin(2.0 * u)  # second harmonic on diagonal
    
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
