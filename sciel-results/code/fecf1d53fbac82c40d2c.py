import numpy as np
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eta = np.zeros_like(distance_matrix, dtype=float)
    for i in range(n):
        # Get distances from i to all other nodes
        dists = distance_matrix[i].copy()
        dists[i] = np.inf # Exclude self
        # Get ranks
        # argsort returns indices of sorted elements
        # We want rank 1 for smallest distance
        sorted_indices = np.argsort(dists)
        for rank, j in enumerate(sorted_indices, start=1):
            eta[i, j] = 1.0 / rank
    return eta
