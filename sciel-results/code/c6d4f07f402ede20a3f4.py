"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: context-aware hybrid multiplicative/additive penalty.
    
    - Context-aware multiplicative penalty on tour edges using normalized usage ratio
      (usage/max_usage)^p, which differentiates between heavily and lightly used edges.
    - Light additive global penalty on all edges based on usage (exploration).
    - Small multiplicative penalty on frequently-used non-tour edges.
    - Adaptive coefficient that grows with round number.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n < 2:
        return edge_distance.copy()

    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)
    tour = local_opt_tour

    # Compute mean distance for scaling
    triu_idx = np.triu_indices(n, k=1)
    mean_dist = np.mean(D[triu_idx])
    if mean_dist < 1e-12:
        mean_dist = 1.0

    # Infer round number from max usage
    global_max = float(U.max()) if U.size > 0 else 0.0
    round_num = max(1, int(round(global_max)))

    # Adaptive multiplicative coefficient: grows with round
    # Round 1: ~0.20, Round 10: ~0.40, Round 20: ~0.55
    a = min(0.6, 0.20 + 0.025 * (round_num - 1))

    # Power for sublinear scaling of usage ratio
    p = 0.6

    # Build the modified distance matrix (start with true distances)
    modified = D.copy()

    # Light additive global penalty on all edges based on usage
    # This gently encourages exploration of unused edges
    sat = U / (1.0 + U)  # soft-saturating, in [0, 1)
    global_scale = 0.02 * mean_dist
    modified += global_scale * sat

    # Context-aware multiplicative penalty on tour edges
    # Collect tour edges and their usages
    tour_edges = []
    tour_usages = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_usages.append(U[u, v])

    tour_usages_arr = np.array(tour_usages, dtype=np.float64)
    max_tour_usage = float(tour_usages_arr.max()) if len(tour_usages_arr) > 0 else 1.0
    if max_tour_usage < 1e-9:
        max_tour_usage = 1.0

    # Apply context-aware multiplicative penalty to tour edges
    for (u, v), usage in zip(tour_edges, tour_usages):
        frac = usage / max_tour_usage
        # Sublinear scaling: edges with moderate usage get relatively more penalty
        # than edges with very high usage (when p < 1)
        r = 1.0 + a * (frac ** p)
        modified[u, v] *= r
        modified[v, u] *= r

    # Small multiplicative penalty on frequently-used non-tour edges
    # This discourages re-forming past local optima
    global_mult_a = 0.03 * (1.0 + 0.03 * np.log1p(round_num))
    soft_global = U / (1.0 + U)
    
    # Mask: non-tour, non-diagonal
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    where_global = mask & (soft_global > 1e-12)
    r_global = 1.0 + global_mult_a * soft_global
    modified = np.where(where_global, modified * r_global, modified)

    # Ensure symmetry
    modified = (modified + modified.T) * 0.5

    # Ensure finite
    modified = np.nan_to_num(modified, nan=0.0,
                             posinf=np.finfo(np.float64).max / 10,
                             neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(modified, 0.0)

    return modified
# EVOLVE-BLOCK-END
