"""Symbolic regression with 1 input(s) - Radial basis + polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 basis functions on [-2, 2].
    
    Basis functions:
      0: 1 (constant)
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: exp(-x^2)          (Gaussian at 0)
      6: exp(-(x-1)^2)      (Gaussian at +1)
      7: exp(-(x+1)^2)      (Gaussian at -1)
      8: exp(-(x-0.5)^2)    (Gaussian at +0.5)
      9: exp(-(x+0.5)^2)    (Gaussian at -0.5)
    
    The Gaussian RBF terms allow capturing local peaks/valleys at various
    locations, while the polynomial terms capture global trends.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    
    # Gaussian RBF terms at different centers
    b5 = np.exp(-x ** 2)
    b6 = np.exp(-(x - 1.0) ** 2)
    b7 = np.exp(-(x + 1.0) ** 2)
    b8 = np.exp(-(x - 0.5) ** 2)
    b9 = np.exp(-(x + 0.5) ** 2)
    
    # Linear combination
    y = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
         params[3] * b3 + params[4] * b4 +
         params[5] * b5 + params[6] * b6 + params[7] * b7 +
         params[8] * b8 + params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
