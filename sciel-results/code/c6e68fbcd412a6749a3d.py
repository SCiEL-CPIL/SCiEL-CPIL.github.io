"""Symbolic regression for a nonlinear oscillator using RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Fixed Radial Basis Function parameters
# Input ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
# We define 10 centers in a grid pattern
CX = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5])
CV = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
SIGMA = 0.35

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict acceleration using a fixed RBF basis.
    The function is linear in `params`, ensuring the optimization problem
    solved by the evaluator (BFGS) is convex and well-behaved.
    """
    # Ensure x and v are 1D arrays
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    
    # Calculate distances to centers
    # dx, dv shape: (N, 10)
    dx = x[:, np.newaxis] - CX[np.newaxis, :]
    dv = v[:, np.newaxis] - CV[np.newaxis, :]
    
    dist_sq = dx**2 + dv**2
    
    # Gaussian basis functions
    # weights shape: (N, 10)
    weights = np.exp(-dist_sq / (2.0 * SIGMA**2))
    
    # Linear combination of basis functions
    pred = weights @ params
    
    return pred
# EVOLVE-BLOCK-END
