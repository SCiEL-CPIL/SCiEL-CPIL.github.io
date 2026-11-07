"""Symbolic regression with polynomial and sinusoidal basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using 10 parameters as coefficients of basis functions.
    
    Basis functions:
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: sin(2x)
    8: cos(2x)
    9: sin(3x)
    """
    p = params
    result = (
        p[0] * np.ones_like(x) +
        p[1] * x +
        p[2] * x**2 +
        p[3] * x**3 +
        p[4] * x**4 +
        p[5] * np.sin(x) +
        p[6] * np.cos(x) +
        p[7] * np.sin(2 * x) +
        p[8] * np.cos(2 * x) +
        p[9] * np.sin(3 * x)
    )
    return result
# EVOLVE-BLOCK-END
