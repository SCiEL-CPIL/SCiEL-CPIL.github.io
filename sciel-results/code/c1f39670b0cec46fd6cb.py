"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Normalized quadratic polynomial basis in 3D.
    
    Inputs are mapped from [0.5, 2] to [-1, 1] to improve conditioning.
    
    Basis functions (10 total):
    0: 1
    1: u
    2: v
    3: w
    4: u^2
    5: v^2
    6: w^2
    7: u*v
    8: u*w
    9: v*w
    
    Where u = 2*(x - 1.25), v = 2*(y - 1.25), w = 2*(z - 1.25)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Normalize inputs to [-1, 1]
    # Center is 1.25, range is 1.5 (from 0.5 to 2.0)
    # u = (x - 1.25) / 0.75 -> maps [0.5, 2] to [-1, 1]
    # Alternatively, simple scaling: u = 2 * (x - 1.25) maps [0.5, 2] to [-1.5, 1.5]
    # Let's use standard normalization to [-1, 1] for better BFGS behavior.
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Extract parameters
    p = params
    
    # Compute basis terms
    b0 = 1.0
    b1 = u
    b2 = v
    b3 = w
    b4 = u * u
    b5 = v * v
    b6 = w * w
    b7 = u * v
    b8 = u * w
    b9 = v * w
    
    # Linear combination
    return p[0]*b0 + p[1]*b1 + p[2]*b2 + p[3]*b3 + p[4]*b4 + p[5]*b5 + p[6]*b6 + p[7]*b7 + p[8]*b8 + p[9]*b9
# EVOLVE-BLOCK-END
