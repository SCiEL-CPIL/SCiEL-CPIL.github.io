"""Symbolic regression using a piecewise linear basis (Hinge functions)."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a piecewise linear basis with 9 internal knots.
    The basis consists of 1 constant term and 9 hinge (ReLU) functions.
    This allows the model to fit any continuous piecewise linear function
    with knots at the specified locations.
    
    params (length 10):
    params[0]: constant term
    params[1]: slope change at knot 1
    ...
    params[9]: slope change at knot 9

    The function is:
        f(x) = p0 + p1 * max(0, x - k0) + ... + p9 * max(0, x - k8)
    
    Knots are evenly spaced in [-2, 2]:
    k_i = -2 + (i+1) * 0.4 for i=0..8
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 9 internal knots evenly spaced in [-2, 2]
    # Spacing is 4.0 / 10 = 0.4
    # Knots at -1.6, -1.2, ..., 1.6
    knots = np.array([-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6], dtype=np.float64)
    
    # Compute basis functions
    # b[0] = 1
    # b[i] = max(0, x - knots[i-1]) for i=1..9
    
    # We can compute this efficiently using broadcasting
    # x shape: (N, 1) or (N,)
    # knots shape: (9,)
    
    x_reshaped = x.reshape(-1, 1)
    hinges = np.maximum(0.0, x_reshaped - knots)
    
    # Construct the basis matrix B of shape (N, 10)
    # Column 0 is ones, columns 1-9 are the hinges
    ones = np.ones((x_reshaped.shape[0], 1))
    B = np.hstack((ones, hinges))
    
    # Compute result: B @ params
    # params shape: (10,)
    result = B.dot(params)
    
    return result.flatten()
# EVOLVE-BLOCK-END
