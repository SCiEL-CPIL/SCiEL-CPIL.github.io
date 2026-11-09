"""Guided local search penalty for TSP."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    # Scale factor: use median of non-zero distances for robust scaling
    flat = edge_distance.flatten()
    nonzero = flat[flat > 0]
    if len(nonzero) > 0:
        scale = np.median(nonzero)
    else:
        scale = 1.0

    # Penalty for edges in the current local-opt tour (strong, to escape this basin)
    tour_penalty = np.zeros_like(edge_distance)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_penalty[a, b] += 1.0
        tour_penalty[b, a] += 1.0

    # Strong penalty on current tour edges
    alpha_tour = 1.5 * scale

    # Penalty on historically-used edges, with saturation to avoid blow-up
    # Normalize usage: divide by (1 + max) so it's in [0, 1]
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / (1.0 + max_used)
    else:
        norm_used = np.zeros_like(edge_n_used)

    # Moderate penalty on historical usage
    alpha_hist = 0.5 * scale

    # Combine: base distance + tour penalty + historical penalty
    penalized = edge_distance.copy()
    penalized += alpha_tour * tour_penalty
    penalized += alpha_hist * norm_used

    # Ensure non-negative and finite
    penalized = np.maximum(penalized, 0.0)
    penalized = np.nan_to_num(penalized, nan=0.0, posinf=np.finfo(float).max / 2, neginf=0.0)

    # Keep diagonal zero (self-loops shouldn't matter, but be safe)
    np.fill_diagonal(penalized, 0.0)

    return penalized
# EVOLVE-BLOCK-END
