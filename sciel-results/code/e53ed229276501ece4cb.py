"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier Series approach
    # Maps x in [-2, 2] to t in [-pi, pi]
    t = np.pi / 2.0 * x
    
    # Extract parameters
    a0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    a3, b3 = params[5], params[6]
    a4, b4 = params[7], params[8]
    a5 = params[9]
    
    # Compute harmonics
    # y = a0 + sum(ak*cos(k*t) + bk*sin(k*t))
    
    # Vectorized calculation
    # To be efficient and clear
    y = np.full_like(x, a0, dtype=float)
    
    y += a1 * np.cos(t) + b1 * np.sin(t)
    y += a2 * np.cos(2 * t) + b2 * np.sin(2 * t)
    y += a3 * np.cos(3 * t) + b3 * np.sin(3 * t)
    y += a4 * np.cos(4 * t) + b4 * np.sin(4 * t)
    y += a5 * np.cos(5 * t)
    
    return y
# EVOLVE-BLOCK-END
