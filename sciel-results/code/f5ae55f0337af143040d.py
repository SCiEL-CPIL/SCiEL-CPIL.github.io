"""Symbolic regression with 2 inputs: Fixed-center Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0

# Precompute centers for the RBF basis
# 3x3 grid in [-1, 1]
_X_CENTERS = [-1.0, 0.0, 1.0]
_Y_CENTERS = [-1.0, 0.0, 1.0]

# Flatten the grid into a list of (cx, cy) pairs
_CENTERS = []
for cy in _Y_CENTERS:
    for cx in _X_CENTERS:
        _CENTERS.append((cx, cy))

# 9 centers + 1 bias = 10 parameters
# Width parameter for Gaussian RBF
_RBF_WIDTH = 2.0 

def _rbf_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the RBF feature matrix.
    Returns an (n, 10) matrix where the first 9 columns are RBFs centered at the grid points,
    and the 10th column is 1 (bias).
    """
    n = x.shape[0]
    F = np.zeros((n, 10), dtype=np.float64)
    
    # Compute RBFs
    for i, (cx, cy) in enumerate(_CENTERS):
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        F[:, i] = np.exp(-_RBF_WIDTH * dist_sq)
        
    # Bias term
    F[:, 9] = 1.0
    
    return F

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _rbf_features(x, y)
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    F = _rbf_features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
