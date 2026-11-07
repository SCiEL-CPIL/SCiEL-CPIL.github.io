"""Rewritten program: richer basis with polynomial + Fourier + rational terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: combination of polynomial, Fourier, and rational basis functions.
    
    params[0]: constant
    params[1]: x
    params[2]: x^2
    params[3]: x^3
    params[4]: x^4
    params[5]: sin(x)
    params[6]: cos(x)
    params[7]: sin(2x)
    params[8]: x / (1 + x^2)
    params[9]: 1 / (1 + x^2)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Fourier terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    
    # Rational terms
    denom = 1.0 + x2
    rational1 = x / denom
    rational2 = 1.0 / denom
    
    return (params[0] +
            params[1] * x +
            params[2] * x2 +
            params[3] * x3 +
            params[4] * x4 +
            params[5] * sin_x +
            params[6] * cos_x +
            params[7] * sin_2x +
            params[8] * rational1 +
            params[9] * rational2)
# EVOLVE-BLOCK-END
