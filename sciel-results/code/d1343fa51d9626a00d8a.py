"""
Symbolic regression: hybrid polynomial and trigonometric basis with 10 features.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, corresponding to:
        1, x, y, x^2, y^2, xy, sin(x), sin(y), cos(x), cos(y)
    
    This basis includes low-order polynomials and trigonometric terms
    to capture a wider range of functional forms.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute predictions
    pred = (
        p0 * np.ones_like(x) +
        p1 * x +
        p2 * y +
        p3 * x**2 +
        p4 * y**2 +
        p5 * x * y +
        p6 * np.sin(x) +
        p7 * np.sin(y) +
        p8 * np.cos(x) +
        p9 * np.cos(y)
    )
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something went wrong
        pred = p0 * np.ones_like(x) + p1 * x + p2 * y
    
    return pred
# EVOLVE-BLOCK-END
