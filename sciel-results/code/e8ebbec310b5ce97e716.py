"""Guided local search penalty rule for the TSP – two-tier penalty with tour-edge exclusion."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Two-tier penalty:
    1. Current tour edges get a large penalty to force 2-opt to explore
       a diverse neighborhood (distance-preserving edge exclusion).
    2. Non-tour edges get a soft geometrically-aware penalty based on
       usage count and rank among incident edges.
    
    The tour-edge penalty decays over rounds; early rounds break more edges.
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

    # --- Compute current tour length and edges ---
    tour_len = 0.0
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_len += D[u, v]

    # --- Infer round number ---
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based

    # --- Tier 1: Large penalty for current tour edges ---
    # Scale relative to max distance to ensure 2-opt avoids these edges
    max_dist = D.max()
    if max_dist > 0:
        tour_edge_penalty = 10.0 * max_dist
    else:
        tour_edge_penalty = 10.0

    result = D.copy()

    # Apply large penalty to tour edges
    for u, v in tour_edges:
        result[u, v] += tour_edge_penalty
        result[v, u] += tour_edge_penalty

    # --- Tier 2: Soft geometrically-aware penalty for non-tour edges ---
    # Annealed penalty coefficient
    k = 1.0 - 0.03 * min(round_num, 20)
    k = max(k, 0.2)

    # Compute geometric rank factor for each edge
    rank_factor = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        row = D[i, :].copy()
        row[i] = np.inf
        order = np.argsort(row)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        ranks[i] = 0.0
        rf = 1.0 / (1.0 + ranks)
        rank_factor[i, :] = rf
        rank_factor[:, i] = rf
    
    rank_factor = (rank_factor + rank_factor.T) * 0.5

    # Soft usage: u/(1+u) for bounded saturation
    soft_U = U / (1.0 + U)

    np.fill_diagonal(rank_factor, 0.0)
    np.fill_diagonal(soft_U, 0.0)
    
    gap = tour_len / max(len(tour), 1)
    
    # Soft penalty for non-tour edges
    soft_penalty = k * gap * soft_U * rank_factor

    # Add soft penalty to non-tour edges only
    # Create a mask for tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    for u, v in tour_edges:
        tour_mask[u, v] = True
        tour_mask[v, u] = True
    
    # Apply soft penalty only where not already penalized heavily
    soft_penalty = np.where(tour_mask, 0.0, soft_penalty)
    result += soft_penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
