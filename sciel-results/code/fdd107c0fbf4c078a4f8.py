"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a Fourier series.
    This approach is effective if the hidden function has periodic or oscillatory components.
    params: array of length 10 [a0, a1, b1, a2, b2, a3, b3, a4, b4, a5]
    y = a0 + sum_{k=1}^{4} (a_k sin(kx) + b_k cos(kx)) + a5 sin(5x)
    """
    # DC component
    y = np.full_like(x, params[0], dtype=float)
    
    # k=1
    y += params[1] * np.sin(x)
    y += params[2] * np.cos(x)
    
    # k=2
    y += params[3] * np.sin(2.0 * x)
    y += params[4] * np.cos(2.0 * x)
    
    # k=3
    y += params[5] * np.sin(3.0 * x)
    y += params[6] * np.cos(3.0 * x)
    
    # k=4
    y += params[7] * np.sin(4.0 * x)
    y += params[8] * np.cos(4.0 * x)
    
    # k=5 (sine only to use 10th param)
    y += params[9] * np.sin(5.0 * x)
    
    return y
# EVOLVE-BLOCK-END
