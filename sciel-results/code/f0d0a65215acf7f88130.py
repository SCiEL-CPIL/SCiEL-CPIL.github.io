# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    t0 = time.time()
    n = distance_matrix.shape[0]
    
    # Ensure we work with a copy
    d = distance_matrix.astype(np.float64).copy()
    
    # Set diagonal to infinity for neighbor finding
    np.fill_diagonal(d, np.inf)
    
    # Find k-nearest neighbors for each node (k=4 gives good local structure)
    k = 4
    # For each node, get indices of k nearest neighbors
    # Use argpartition for efficiency
    nn_indices = np.argpartition(d, k, axis=1)[:, :k]
    nn_distances = np.take_along_axis(d, nn_indices, axis=1)
    
    # Build a "nearest-neighbor edge" matrix: 1 if j is among i's k-nearest
    nn_edge = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in nn_indices[i]:
            nn_edge[i, j] = 1.0
            nn_edge[j, i] = 1.0  # symmetrize
    
    # Base heuristic: inverse distance with sharpened exponent
    # Use 1/d^2 to make short edges much more attractive
    eps = 1e-9
    d_safe = d + eps
    base = 1.0 / (d_safe ** 2.0)
    
    # Strongly boost nearest-neighbor edges
    # This creates a sparse, high-signal heuristic
    nn_boost = 1.0 + 10.0 * nn_edge  # 11x boost for NN edges
    
    # Additional boost for mutual nearest neighbors (both i in j's kNN and j in i's kNN)
    mutual_nn = nn_edge * nn_edge.T  # 1 where both directions are NN
    mutual_boost = 1.0 + 5.0 * mutual_nn
    
    # Combine
    result = base * nn_boost * mutual_boost
    
    # Ensure positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    # Symmetrize
    result = 0.5 * (result + result.T)
    
    # Final safety
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
