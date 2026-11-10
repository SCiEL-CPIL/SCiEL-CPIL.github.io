"""Symbolic regression with 2 inputs: Gaussian RBF basis with strategic centers."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0

# Precompute RBF centers and width
# Use a mix of global (wide) and local (narrow) RBFs
# Centers chosen to cover the [-1,1]^2 domain well
_CENTERS = np.array([
    [0.0, 0.0],      # center
    [-0.5, -0.5],    # corner
    [0.5, 0.5],      # corner
    [-0.5, 0.5],     # corner
    [0.5, -0.5],     # corner
    [0.0, 0.7],      # top
    [0.0, -0.7],     # bottom
    [0.7, 0.0],      # right
    [-0.7, 0.0],     # left
    [0.0, 0.0],      # duplicate center with different width
])

# Widths: mix of wide (global) and narrow (local) RBFs
_WIDTHS = np.array([
    1.0,    # wide - captures global structure
    0.5,    # medium
    0.5,    # medium
    0.5,    # medium
    0.5,    # medium
    0.4,    # medium-narrow
    0.4,    # medium-narrow
    0.4,    # medium-narrow
    0.4,    # medium-narrow
    0.2,    # narrow - captures local features
])


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix using Gaussian RBF basis.
    
    Each feature is exp(-||[x,y] - c_k||^2 / (2 * sigma_k^2)) where c_k is the k-th center
    and sigma_k is the k-th width. This provides a flexible, local basis that can
    approximate a wide variety of smooth functions.
    """
    n = x.shape[0]
    
    # Stack x and y into (n, 2) array
    pts = np.column_stack([x, y])  # (n, 2)
    
    # Compute squared distances from each point to each center: (n, 10)
    # Use broadcasting: pts (n,1,2) - centers (1,10,2)
    diff = pts[:, np.newaxis, :] - _CENTERS[np.newaxis, :, :]  # (n, 10, 2)
    sq_dist = np.sum(diff * diff, axis=2)  # (n, 10)
    
    # Compute RBF values: exp(-sq_dist / (2 * sigma^2))
    # sigma^2 is (10,), so broadcast: (n, 10) / (1, 10)
    sigma2 = _WIDTHS ** 2  # (10,)
    F = np.exp(-sq_dist / (2.0 * sigma2[np.newaxis, :]))  # (n, 10)
    
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
