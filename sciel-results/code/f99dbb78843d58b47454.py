"""Symbolic regression with linear basis of diverse normalized functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear parametric model with 10 diverse basis functions.
    
    The model is linear in parameters, which makes the optimization problem
    convex (least squares). BFGS will reliably find the global optimum.
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: sin(2x)
    8: cos(2x)
    9: exp(-x^2)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Build basis matrix
    # Shape: (N, 10)
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        np.sin(x),
        np.cos(x),
        np.sin(2 * x),
        np.cos(2 * x),
        np.exp(-x**2)
    ])
    
    # Linear combination
    y = basis @ params
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
