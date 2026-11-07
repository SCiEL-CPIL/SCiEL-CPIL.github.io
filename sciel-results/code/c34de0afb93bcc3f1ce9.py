"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a mixed basis of low-order polynomials and low-order Fourier terms.
    This structural change aims to capture both smooth polynomial trends and
    periodic/oscillatory components that pure Chebyshev polynomials might
    approximate poorly with limited degrees.
    
    Basis terms:
    0: 1 (Constant)
    1: x (Linear x)
    2: y (Linear y)
    3: x^2 - 0.5 (Centered Quadratic x)
    4: y^2 - 0.5 (Centered Quadratic y)
    5: sin(pi * x) (Fourier x)
    6: sin(pi * y) (Fourier y)
    7: cos(pi * x) (Fourier x)
    8: cos(pi * y) (Fourier y)
    9: x * y (Interaction)
    """
    # Polynomial terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x**2 - 0.5
    term4 = y**2 - 0.5
    term9 = x * y
    
    # Fourier terms
    # Note: pi * x maps [-1, 1] to [-pi, pi]
    term5 = np.sin(np.pi * x)
    term6 = np.sin(np.pi * y)
    term7 = np.cos(np.pi * x)
    term8 = np.cos(np.pi * y)
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
