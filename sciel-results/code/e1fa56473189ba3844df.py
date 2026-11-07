"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0, 1)."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        i_val = i
        while i_val > 0:
            f /= base
            r += f * (i_val % base)
            i_val //= base
        result.append(r)
    return np.array(result)

def _generate_rbf_centers(n_centers):
    """Generate deterministic RBF centers using Halton sequence over [-1, 1]."""
    x_halton = _halton_sequence(n_centers, 2)
    y_halton = _halton_sequence(n_centers, 3)
    # Map from [0, 1) to [-1, 1]
    x_centers = 2.0 * x_halton - 1.0
    y_centers = 2.0 * y_halton - 1.0
    return np.column_stack([x_centers, y_centers])

# Precompute centers
N_CENTERS = 10
CENTERS = _generate_rbf_centers(N_CENTERS)
SIGMA = 0.8  # Width of the Gaussian kernel

def rbf_basis_2d(x, y):
    """
    Returns an array of shape (N, 10) with the 10 RBF basis functions.
    """
    N = len(x)
    points = np.column_stack([x, y]) # (N, 2)
    
    # Compute squared distances from each point to each center
    # points: (N, 1, 2), CENTERS: (1, 10, 2)
    dists_sq = np.sum((points[:, np.newaxis, :] - CENTERS[np.newaxis, :, :])**2, axis=2) # (N, 10)
    
    # Apply Gaussian kernel
    basis = np.exp(-dists_sq / (2 * SIGMA**2))
    
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if params are valid
    if not np.all(np.isfinite(params)):
        return np.zeros(len(x))
        
    # Compute basis
    basis = rbf_basis_2d(x, y)
    
    # Predict
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(len(x))
        
    return pred
# EVOLVE-BLOCK-END
