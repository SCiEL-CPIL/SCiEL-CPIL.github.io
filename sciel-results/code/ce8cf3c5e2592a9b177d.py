"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Fourier series basis.
    This allows the model to fit functions with periodic or oscillatory components.
    
    params: array of length 10
    Basis:
    p[0]: 1
    p[1]: cos(1*x)
    p[2]: sin(1*x)
    p[3]: cos(2*x)
    p[4]: sin(2*x)
    p[5]: cos(3*x)
    p[6]: sin(3*x)
    p[7]: cos(4*x)
    p[8]: sin(4*x)
    p[9]: cos(5*x)
    """
    x = np.asarray(x)
    
    # Initialize prediction with the constant term
    pred = np.full_like(x, params[0], dtype=float)
    
    # Add harmonic components
    # k=1
    pred += params[1] * np.cos(1.0 * x) + params[2] * np.sin(1.0 * x)
    # k=2
    pred += params[3] * np.cos(2.0 * x) + params[4] * np.sin(2.0 * x)
    # k=3
    pred += params[5] * np.cos(3.0 * x) + params[6] * np.sin(3.0 * x)
    # k=4
    pred += params[7] * np.cos(4.0 * x) + params[8] * np.sin(4.0 * x)
    # k=5 (only cosine to use the 10th parameter)
    pred += params[9] * np.cos(5.0 * x)
    
    return pred
# EVOLVE-BLOCK-END
