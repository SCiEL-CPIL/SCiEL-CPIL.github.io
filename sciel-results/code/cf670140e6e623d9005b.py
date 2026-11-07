"""Symbolic regression with 1 input using a rich basis of common functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a linear combination of 10 basis functions to recover an unknown target.
    The basis includes: 1, x, x^2, x^3, sin(x), cos(x), exp(-x), 1/x, log(x), sqrt(x).
    This provides a flexible model that can approximate many common functional forms.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # Build basis functions
    # Ensure x is positive for log and sqrt (x in [0.1, 6] so this is safe)
    x_safe = np.maximum(x, 1e-12)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.exp(-x)
    b7 = 1.0 / x_safe
    b8 = np.log(x_safe)
    b9 = np.sqrt(x_safe)
    
    # Stack basis into matrix (n_points, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B @ params
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
