# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0


def _safe_tanh(z):
    """Numerically stable tanh."""
    return np.tanh(z)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs.
    Uses a flexible basis: linear, quadratic, cubic, and interaction terms,
    plus some non-linear activations to capture complex structure.
    
    The 10 parameters are used in a structured way:
    - p[0:5]: coefficients for a quadratic polynomial basis
    - p[5:10]: coefficients for additional non-linear/interaction features
    """
    n = len(x)
    
    # Basic features
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Quadratic polynomial basis (5 terms)
    # 1, x, y, x^2, y^2
    f0 = np.ones(n)
    f1 = x
    f2 = y
    f3 = x2
    f4 = y2
    
    # Additional features using remaining 5 params
    # Interaction and non-linear terms
    f5 = xy
    f6 = x2 * y
    f7 = x * y2
    f8 = x2 * y2
    f9 = np.tanh(3.0 * (x + y))  # Non-linear activation
    
    # Build feature matrix
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    return features @ params
# EVOLVE-BLOCK-END
