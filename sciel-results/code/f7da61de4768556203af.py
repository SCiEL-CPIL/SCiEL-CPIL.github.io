"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Refined trigonometric basis with fundamental frequencies and triple-sum interaction.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: sin(x)
    2: cos(x)
    3: sin(y)
    4: cos(y)
    5: sin(z)
    6: cos(z)
    7: sin(x + y + z)
    8: sin(x + y)
    9: sin(y + z)
    
    This representation captures periodic dependencies with lower frequencies
    and allows for three-way interactions.
    """
    p = params.flatten()
    
    # Ensure we have at least 10 parameters
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    a4 = p[4]
    a5 = p[5]
    a6 = p[6]
    a7 = p[7]
    a8 = p[8]
    a9 = p[9]
    
    # Precompute trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    sin_z = np.sin(z)
    cos_z = np.cos(z)
    sin_xyz = np.sin(x + y + z)
    sin_xy = np.sin(x + y)
    sin_yz = np.sin(y + z)
    
    # Build the expression
    result = a0
    result += a1 * sin_x
    result += a2 * cos_x
    result += a3 * sin_y
    result += a4 * cos_y
    result += a5 * sin_z
    result += a6 * cos_z
    result += a7 * sin_xyz
    result += a8 * sin_xy
    result += a9 * sin_yz
    
    return result
# EVOLVE-BLOCK-END
