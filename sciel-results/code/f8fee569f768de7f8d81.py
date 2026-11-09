"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Heuristic matrix for ant colony optimization of TSP.
    
    Uses a softened inverse distance (d^-0.8) as the base heuristic,
    combined with a symmetric k-nearest-neighbor structural correction
    to encourage balanced tours and avoid local cycles.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Soften the inverse distance: use d^{-0.8} instead of d^{-1}
    # This reduces the overwhelming pull of the nearest neighbor
    # and allows ants to consider slightly farther nodes, breaking
    # local cycles more effectively.
    d_safe = dm + 1e-9
    base_heuristic = 1.0 / (d_safe ** 0.8)
    
    # Compute a symmetric k-NN structural correction
    # For each pair (i,j), check if j is in i's k-NN and i is in j's k-NN
    # If both, boost the edge; if neither, slightly penalize
    k = 4
    nn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        distances_i = dm[i].copy()
        distances_i[i] = np.inf
        k_nn_idx = np.argsort(distances_i)[:k]
        nn_mask[i, k_nn_idx] = True
    
    # Symmetric mask: edge is "good" if both directions are in k-NN
    sym_mask = nn_mask & nn_mask.T
    
    # Compute a correction factor:
    # - 1.15 if symmetrically in k-NN (encourage these edges)
    # - 0.85 if not symmetrically in k-NN (discourage)
    correction_factor = np.where(sym_mask, 1.15, 0.85)
    
    # Apply correction
    heuristic = base_heuristic * correction_factor
    
    # Ensure diagonal is very small (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so the mean is around 1 for consistency
    mean_val = heuristic[heuristic > 0].mean()
    if mean_val > 0:
        heuristic = heuristic / mean_val
    
    return heuristic
# EVOLVE-BLOCK-END
