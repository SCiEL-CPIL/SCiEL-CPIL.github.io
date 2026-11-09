"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Uses a flexible basis of functions to approximate the target.
    Basis functions (10 total):
    0: constant
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: tanh(x)
    7: exp(-0.5 * x^2)  (Gaussian bump at 0)
    8: exp(-0.5 * (x-1)^2)  (Gaussian bump at 1)
    9: exp(-0.5 * (x+1)^2)  (Gaussian bump at -1)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim > 1:
            x_arr = x_arr.ravel()
        return np.zeros_like(x_arr, dtype=float)
    
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=float)
    
    # Ensure finite params
    p = np.where(np.isfinite(p), p, 1.0)
    
    # Clip extreme parameter values for numerical stability
    p = np.clip(p, -1e4, 1e4)
    
    # Compute basis functions with numerical safety
    x2 = x * x
    basis = np.column_stack([
        np.ones_like(x),       # p0: constant
        x,                      # p1: x
        x2,                     # p2: x^2
        x2 * x,                 # p3: x^3
        np.sin(x),              # p4: sin(x)
        np.cos(x),              # p5: cos(x)
        np.tanh(x),             # p6: tanh(x)
        np.exp(-0.5 * x2),      # p7: Gaussian at 0
        np.exp(-0.5 * (x - 1.0) ** 2),  # p8: Gaussian at 1
        np.exp(-0.5 * (x + 1.0) ** 2),  # p9: Gaussian at -1
    ])
    
    result = basis @ p
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
