"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty: combine additive usage-based penalties
    with a strong repulsion from the current tour edges to break local optima.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy
    penalized = edge_distance.copy()
    
    # 1) Additive penalty based on historical edge usage.
    #    Scale by the median edge distance for robustness.
    median_dist = np.median(edge_distance[np.triu_indices(n, k=1)])
    if median_dist == 0:
        median_dist = 1.0
    
    # Penalty weight: small enough to not distort too much, but accumulates over rounds.
    # Using a factor that grows slightly with usage for stronger repulsion of frequent edges.
    base_weight = 0.12 * median_dist
    # Add a quadratic component for very frequently used edges to break stubborn cycles
    penalty_matrix = base_weight * edge_n_used + 0.02 * median_dist * (edge_n_used ** 2) / (1.0 + edge_n_used.max())
    
    # Ensure symmetry (edge_n_used should be symmetric, but be safe)
    penalty_matrix = (penalty_matrix + penalty_matrix.T) / 2.0
    
    penalized += penalty_matrix
    
    # 2) Strong penalty on edges in the current local-optimal tour.
    #    This forces 2-opt to consider replacing these edges.
    #    Use a penalty that is significant but not infinite, to keep the matrix well-behaved.
    tour_penalty = 0.5 * median_dist
    
    for i in range(n):
        j = local_opt_tour[i]
        if i != j:
            penalized[i, j] += tour_penalty
            penalized[j, i] += tour_penalty
    
    # 3) Optional: apply a mild edge-reversal bias for the most-used tour edges.
    #    For edges with high usage, also penalize slightly less to allow some flexibility,
    #    but the main effect is already captured by the additive penalty above.
    
    # Ensure the matrix is symmetric
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure all values are finite and non-negative
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 0.0)
    
    # Time guard: if we're taking too long, return early (though this function is fast)
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        # Fallback to a simple penalty if something went wrong
        return edge_distance + 0.1 * median_dist * edge_n_used
    
    return penalized
# EVOLVE-BLOCK-END
