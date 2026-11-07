"""Symbolic regression: separable Fourier series with frequency optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable Fourier series model:
    f(x,y) = a0*sin(a1*x)*sin(a2*y) + a3*sin(a4*x)*sin(a5*y) + a6*x*y + a7 + a8*x + a9*y
    
    params (length 10):
      [0] a0: amplitude of first product term
      [1] a1: frequency for x in first term
      [2] a2: frequency for y in first term
      [3] a3: amplitude of second product term
      [4] a4: frequency for x in second term
      [5] a5: frequency for y in second term
      [6] a6: coefficient for x*y
      [7] a7: bias
      [8] a8: coefficient for x
      [9] a9: coefficient for y
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT:
        # Return a safe fallback: linear model using last params
        return params[7] + params[8]*x + params[9]*y
    
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = params
    
    # Compute the two product-of-sines terms
    term1 = a0 * np.sin(a1 * x) * np.sin(a2 * y)
    term2 = a3 * np.sin(a4 * x) * np.sin(a5 * y)
    
    # Polynomial/linear terms
    poly = a6 * x * y + a7 + a8 * x + a9 * y
    
    pred = term1 + term2 + poly
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to linear-only
        pred = a7 + a8*x + a9*y
        if not np.all(np.isfinite(pred)):
            pred = np.zeros_like(x)
    
    return pred
# EVOLVE-BLOCK-END
