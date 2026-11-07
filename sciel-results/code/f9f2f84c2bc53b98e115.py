"""Symbolic regression candidate: Fourier series with 10 coefficients."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form Fourier series model with 10 learnable coefficients.
    
    Basis functions (10 total):
      0: 1 (constant)
      1: x
      2: y
      3: sin(x)
      4: cos(x)
      5: sin(y)
      6: cos(y)
      7: sin(x + y)
      8: cos(x - y)
      9: sin(2*x) * cos(2*y)
    
    This is linear in params, so BFGS will converge to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = params[:10]
    
    # Build basis functions
    f = np.empty((10, x.size), dtype=np.float64)
    f[0] = 1.0
    f[1] = x
    f[2] = y
    f[3] = np.sin(x)
    f[4] = np.cos(x)
    f[5] = np.sin(y)
    f[6] = np.cos(y)
    f[7] = np.sin(x + y)
    f[8] = np.cos(x - y)
    f[9] = np.sin(2.0 * x) * np.cos(2.0 * y)
    
    # Linear combination
    return f @ p
# EVOLVE-BLOCK-END
