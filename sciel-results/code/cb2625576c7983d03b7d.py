"""Structural change: multi-basis symbolic regression candidate."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a flexible basis.
    
    The basis combines:
      - Polynomial terms (1, x, x^2, x^3)
      - Trigonometric terms (sin(x), cos(x), sin(2x), cos(2x))
      - Exponential-like terms (exp(x), exp(-x))
      - Reciprocal terms (1/(1+x^2))
    
    This gives 10 basis functions, matched to the 10 params.
    """
    x = np.asarray(x, dtype=float)
    p = params
    
    # Basis functions
    b0 = np.ones_like(x)          # constant
    b1 = x                         # linear
    b2 = x * x                     # quadratic
    b3 = x * x * x                 # cubic
    b4 = np.sin(x)                 # sine
    b5 = np.cos(x)                 # cosine
    b6 = np.sin(2.0 * x)           # sin(2x)
    b7 = np.cos(2.0 * x)           # cos(2x)
    b8 = 1.0 / (1.0 + x * x)       # rational
    b9 = np.exp(x) - np.exp(-x)    # sinh-like (odd)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
