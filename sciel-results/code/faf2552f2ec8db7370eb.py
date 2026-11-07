"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric harmonic basis with single-variable sine/cosine and pairwise sum interactions.
    """
    p = params
    
    # Basis 0: Constant
    b0 = np.ones_like(x)
    
    # Basis 1-3: Single variable sines
    b1 = np.sin(x)
    b2 = np.sin(y)
    b3 = np.sin(z)
    
    # Basis 4-6: Single variable cosines
    b4 = np.cos(x)
    b5 = np.cos(y)
    b6 = np.cos(z)
    
    # Basis 7-9: Pairwise sum interactions (sine)
    b7 = np.sin(x + y)
    b8 = np.sin(y + z)
    b9 = np.sin(z + x)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
