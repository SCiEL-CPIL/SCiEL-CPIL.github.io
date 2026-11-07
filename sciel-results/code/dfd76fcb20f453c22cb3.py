"""Trigonometric basis for 2D periodic symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric basis using 10 parameters for a 2D periodic function:
    - params[0]: constant (intercept)
    - params[1]: cos(x)
    - params[2]: sin(x)
    - params[3]: cos(y)
    - params[4]: sin(y)
    - params[5]: cos(2*x)
    - params[6]: sin(2*x)
    - params[7]: cos(x+y)
    - params[8]: sin(x+y)
    - params[9]: cos(x-y)
    """
    p = params
    
    px = np.pi * x
    py = np.pi * y
    
    pred = (
        p[0]
        + p[1] * np.cos(px)
        + p[2] * np.sin(px)
        + p[3] * np.cos(py)
        + p[4] * np.sin(py)
        + p[5] * np.cos(2.0 * px)
        + p[6] * np.sin(2.0 * px)
        + p[7] * np.cos(px + py)
        + p[8] * np.sin(px + py)
        + p[9] * np.cos(px - py)
    )
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
