"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: add targeted and global penalties.
    
    - Targeted: penalize edges in the current tour strongly, scaled by usage.
    - Global: penalize all edges lightly, scaled by usage.
    - Both scaled relative to mean edge distance for robustness.
    """
    t0 = time.time()
    n = edge_distance.shape[0]
    mean_dist = np.mean(edge_distance[np.triu_indices(n, k=1)])
    if mean_dist < 1e-12:
        mean_dist = 1.0

    # Soft-saturating usage factor
    u = edge_n_used.astype(np.float64)
    sat = u / (1.0 + u)

    # Build the modified distance matrix
    modified = edge_distance.copy()

    # Global penalty: light penalty on all edges based on usage
    global_scale = 0.05 * mean_dist
    modified += global_scale * sat

    # Targeted penalty: strong penalty on edges in the current tour
    tour_scale = 0.5 * mean_dist
    for i in range(n):
        j = local_opt_tour[i]
        modified[i, j] += tour_scale * sat[i, j]
        modified[j, i] += tour_scale * sat[j, i]

    # Time guard: if we're close to deadline, return immediately
    if time.time() - t0 > 5.0:
        pass  # Already fast enough for n=100

    return modified
# EVOLVE-BLOCK-END
