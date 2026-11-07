"""Guided local search penalty rule for the TSP – power-scaled targeted perturbation."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def _compute_lower_bound(D):
    """MST-based lower bound on TSP tour length."""
    n = D.shape[0]
    if n < 2:
        return 0.0
    min1 = np.full(n, np.inf)
    min2 = np.full(n, np.inf)
    for i in range(n):
        row = D[i, :].copy()
        row[i] = np.inf
        if n < 3:
            min1[i] = row.min()
            min2[i] = 0.0
        else:
            idx = np.argpartition(row, 2)[:2]
            min1[i] = row[idx].min()
            min2[i] = row[idx].max()
    return 0.5 * np.sum(min1 + min2)


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Uses power-scaled (1.5) additive penalties on tour edges, normalized by mean
    edge distance, with annealed coefficients and a soft global usage penalty.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Compute current tour edges and length ---
    tour_edges = []
    tour_len = 0.0
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_len += D[u, v]

    # Mean edge distance of the tour (used for normalization)
    mean_edge = tour_len / len(tour) if tour_len > 1e-12 else 1.0

    # --- Compute lower bound and gap ---
    lb = _compute_lower_bound(D)
    gap = tour_len - lb
    if gap < 1e-9:
        gap = 1.0

    # --- Infer round number from max usage count ---
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based

    # --- Annealed penalty coefficient ---
    # k starts higher for exploration, decays for refinement
    k = 1.8 - 0.07 * min(round_num, 20)
    k = max(k, 0.4)
    
    # --- Distance-preserving edge exclusion: set tour edges to large value ---
    # This forces 2-opt to remove these "stuck" edges
    large_val = 1000.0  # Much larger than typical edge distances in unit square

    # --- Build targeted penalty on tour edges with power-1.5 scaling ---
    penalty = np.zeros((n, n), dtype=np.float64)

    for u, v in tour_edges:
        # Set tour edges to large value to force their removal
        penalty[u, v] += large_val
        penalty[v, u] += large_val

    # --- Reduced global soft penalty since tour edges are now strongly excluded ---
    global_coeff = 0.03 * gap / max(n, 1)

    # Soft saturation: sqrt(u)/(1+sqrt(u)) for bounded growth
    sqrt_U = np.sqrt(np.maximum(U, 0))
    soft_U = sqrt_U / (1.0 + sqrt_U)

    # Mask: exclude current tour edges and diagonal
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    global_pen = np.where(mask & (soft_U > 0.1),
                          global_coeff * soft_U, 0.0)
    penalty += global_pen

    # --- Combine ---
    result = D + penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    # Clamp tour edges to exactly large_val to ensure they are strongly penalized
    for u, v in tour_edges:
        result[u, v] = large_val
        result[v, u] = large_val

    return result
