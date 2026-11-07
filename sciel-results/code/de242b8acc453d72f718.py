"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a flexible basis expansion.
    
    We use a combination of polynomial terms, localized Gaussian bumps, and
    trigonometric harmonics to capture a wide variety of functional forms.
    
    params (length 10) are linear coefficients for the basis functions.
    The basis functions are:
        1, x, x^2, exp(-(x-1)^2), exp(-(x-3)^2), exp(-(x-5)^2),
        sin(x), cos(x), sin(2x), cos(2x)
    
    This linear-in-parameters model allows BFGS to efficiently find the best
    linear combination. The Gaussian bumps allow for local features, while
    polynomials and trig terms handle global trends and oscillations.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Build basis matrix
    b1 = np.ones_like(x)
    b2 = x
    b3 = x ** 2
    
    # Gaussian bumps centered at 1, 3, 5 with width 1
    b4 = np.exp(-(x - 1.0) ** 2)
    b5 = np.exp(-(x - 3.0) ** 2)
    b6 = np.exp(-(x - 5.0) ** 2)
    
    # Trigonometric terms (fundamental and 2nd harmonic)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.sin(2.0 * x)
    b10 = np.cos(2.0 * x)
    
    # Linear combination
    result = (
        params[0] * b1 +
        params[1] * b2 +
        params[2] * b3 +
        params[3] * b4 +
        params[4] * b5 +
        params[5] * b6 +
        params[6] * b7 +
        params[7] * b8 +
        params[8] * b9 +
        params[9] * b10
    )
    
    return result
# EVOLVE-BLOCK-END
