"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 325.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Identify edges in the current local optimum tour
    tour_edges = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = 1.0
        tour_edges[j, i] = 1.0
        tour_edges[i, k] = 1.0
        tour_edges[k, i] = 1.0

    # Get usage counts for tour edges
    edge_counts = edge_n_used * tour_edges

    # Adaptive penalty: stronger for frequently used edges
    # Use a penalty that scales with usage count but is bounded
    max_count = edge_n_used.max() if edge_n_used.size > 0 else 1.0
    if max_count == 0:
        max_count = 1.0

    # Penalty parameter: increase with usage but keep it reasonable
    # For edges used many times, apply larger penalty to encourage diversification
    penalty_factor = 0.5 + 0.5 * (edge_counts / (1.0 + max_count))

    # Apply penalty only to tour edges
    result = edge_distance.copy()
    result[tour_edges > 0] *= (1.0 + penalty_factor[tour_edges > 0])

    # Ensure all values are finite
    result[~np.isfinite(result)] = edge_distance[~np.isfinite(result)] if np.any(~np.isfinite(edge_distance)) else 1.0

    return result
# EVOLVE-BLOCK-END
