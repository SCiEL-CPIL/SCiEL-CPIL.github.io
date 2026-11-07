"""Guided local search penalty rule for the TSP – linear un-normalized penalty."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.

    Strategy:
    - Uses un-normalized usage counts to allow penalties to grow linearly
      with the number of times an edge has been used in past local optima.
    - This provides a stronger perturbation in later rounds to escape
      persistent local optima, addressing the limitation of normalized
      penalties which saturate.
    - Applies a primary penalty to edges in the current tour.
    - Applies a secondary, smaller penalty to other frequently used edges
      to discourage the re-formation of similar sub-tours.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    # --- Extract tour edges and compute scale ---
    tour_edges = []
    tour_lens = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_lens.append(edge_distance[u, v])

    avg_len = float(np.mean(tour_lens)) if tour_lens else 1.0
    if avg_len < 1e-9:
        avg_len = 1.0

    # --- Build penalty matrix ---
    penalty = np.zeros((n, n), dtype=np.float64)

    # Coefficients for linear un-normalized penalty
    # C1 controls the strength of the kick for current tour edges
    # C2 controls the strength of the kick for other historically used edges
    C1 = 0.1
    C2 = 0.02

    # 1. Penalty for edges in the current tour
    # Using raw usage count (un-normalized) to allow strong growth over rounds
    for u, v in tour_edges:
        usage = float(edge_n_used[u, v])
        ep = C1 * avg_len * usage
        penalty[u, v] += ep
        penalty[v, u] += ep

    # 2. Global penalty for frequently-used edges NOT in the current tour
    # This helps break cycles of similar solutions
    usage_mat = edge_n_used.astype(np.float64)
    
    # Mask: True where NOT a current-tour edge (and not diagonal)
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    # Apply global penalty only where mask is True and usage > 0
    # Using raw usage count here as well
    global_pen = np.where(mask & (usage_mat > 0),
                          C2 * avg_len * usage_mat, 0.0)
    penalty += global_pen

    # --- Combine ---
    result = edge_distance + penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
