"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    We build a rich basis of smooth functions of (x, y) and let BFGS fit the coefficients.
    
    Basis design:
    - Constant
    - Linear: x, y
    - Quadratic: x^2, y^2, x*y
    - Trigonometric: sin(x), sin(y), cos(x+y)
    - Smooth interaction: tanh(x*y)
    
    We scale basis values to be O(1) on [-1,1]^2 so BFGS from ones works well.
    """
    # Time guard
    t0 = time.time()
    
    # Ensure x, y are 1-D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Build basis functions (each returns an array of same shape as x)
    # We want 10 basis functions for 10 params
    b0 = np.ones_like(x)                          # constant
    b1 = x                                        # linear x
    b2 = y                                        # linear y
    b3 = x * x                                    # x^2
    b4 = y * y                                    # y^2
    b5 = x * y                                    # x*y
    b6 = np.sin(np.pi * x)                        # sin(pi*x) - smooth, O(1) on [-1,1]
    b7 = np.sin(np.pi * y)                        # sin(pi*y)
    b8 = np.cos(np.pi * (x + y))                  # cos(pi*(x+y)) - interaction
    b9 = np.tanh(x * y)                           # smooth bilinear-like interaction
    
    # Stack basis: shape (10, n)
    B = np.vstack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Predictions: B @ params, shape (n,)
    pred = B @ params
    
    # Safety: if anything went wrong, return something finite
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    # Time guard check
    if time.time() - t0 > 100:
        return pred
    
    return pred
# EVOLVE-BLOCK-END
