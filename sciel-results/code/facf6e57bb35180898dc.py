"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses a combination of:
    1. Base inverse distance (1/d)
    2. Nearest-neighbor edge boost (k=4 NN edges get 11x, mutual NN get extra 5x)
    3. Soft-min surrogate to smooth the selection pressure
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    # Work with a copy
    dm = distance_matrix.astype(np.float64).copy()
    
    # Replace zero distances (self-loops) with a small value for computation
    # but we'll mask them out at the end
    diag_mask = np.eye(n, dtype=bool)
    dm_safe = dm.copy()
    dm_safe[diag_mask] = 1.0  # placeholder, will be zeroed later
    
    # Base heuristic: inverse distance
    base = 1.0 / (dm_safe + 1e-9)
    base[diag_mask] = 0.0
    
    # Find k-nearest neighbors for each node
    k = min(4, n - 1)
    # For each row, find the k smallest distances (excluding self)
    nn_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = dm_safe[i].copy()
        row[i] = np.inf  # exclude self
        # Get indices of k smallest
        idx = np.argpartition(row, k)[:k]
        nn_edges[i, idx] = True
    
    # Identify mutual nearest neighbors
    mutual = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if nn_edges[i, j] and nn_edges[j, i]:
                mutual[i, j] = True
                mutual[j, i] = True
    
    # Apply boosts
    # NN edges get 11x, mutual NN get additional 5x (total 11*5=55x for mutual)
    boosted = base.copy()
    boosted[nn_edges] *= 11.0
    boosted[mutual] *= 5.0  # additional multiplier on top of 11x
    
    # Ensure diagonal is zero
    boosted[diag_mask] = 0.0
    
    # Clip to ensure positivity, then set non-positive to 1e-9
    boosted = np.where(boosted > 0, boosted, 1e-9)
    
    # Normalize to keep values in a reasonable range
    # Scale so that the max is around 1.0 for numerical stability
    max_val = np.max(boosted)
    if max_val > 0:
        boosted = boosted / max_val
    
    return boosted
# EVOLVE-BLOCK-END
