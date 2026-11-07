"""Multi-scale fixed-basis linear model for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 fixed basis functions.
    All 10 params are linear coefficients, making the loss convex in params,
    which ensures BFGS converges reliably from the all-ones start.
    
    Basis functions chosen to span polynomials, trigonometric, and localized
    features over x in [-2, 2]:
      0: 1                  (constant)
      1: x                  (linear)
      2: x^2                (quadratic)
      3: x^3                (cubic)
      4: x^4                (quartic)
      5: sin(pi*x)          (trig, period 2)
      6: cos(pi*x)          (trig, period 2)
      7: sin(2*pi*x)        (trig, period 1)
      8: cos(2*pi*x)        (trig, period 1)
      9: exp(-x^2)          (Gaussian bump at 0, localized)
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Build basis matrix: shape (n, 10)
    # Use efficient vectorized operations
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    basis = np.empty((n, 10), dtype=float)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x2
    basis[:, 3] = x3
    basis[:, 4] = x4
    basis[:, 5] = np.sin(np.pi * x)
    basis[:, 6] = np.cos(np.pi * x)
    basis[:, 7] = np.sin(2.0 * np.pi * x)
    basis[:, 8] = np.cos(2.0 * np.pi * x)
    basis[:, 9] = np.exp(-x2)
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
