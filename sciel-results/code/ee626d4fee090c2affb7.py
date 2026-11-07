"""Guided local search penalty rule for the TSP – structural edge-removal penalty."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Structural penalty: make frequently-used tour edges effectively unusable
    by setting their distance to a large value, forcing 2-opt to find
    alternative connections. This is a more aggressive, binary-like penalty
    compared to smooth multiplicative penalties.

    The key structural change: instead of smoothly increasing edge costs,
    we threshold edges based on usage count. Edges that appear in the tour
    AND have been used above a threshold get a very large penalty, making
    2-opt almost certainly remove them. This creates a hard constraint-like
    effect that forces escape from persistent local optima.

    Strategy:
    - For each edge in the current tour, check its usage count.
    - If usage >= threshold (which grows with round number), set the edge
      distance to a large value (but finite) so 2-opt will avoid it.
    - Edges below the threshold get a mild penalty to encourage gradual
      improvement.
    - This creates a 'burning' effect: once an edge is used enough times,
      it becomes effectively forbidden, forcing the tour to restructure.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    # Work in float64
    D = np.asarray(edge_distance, dtype=np.float64).copy()
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Extract tour edges ---
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))

    # Infer round number from max usage count
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based round estimate

    # Compute average edge length for scaling
    tour_lens = [D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))]
    avg_len = float(np.mean(tour_lens)) if tour_lens else 1.0
    if avg_len < 1e-9:
        avg_len = 1.0

    # Large penalty value: make edge effectively unusable
    # Scale with the max distance in the matrix to ensure it's large enough
    max_d = float(D.max()) if D.size > 0 else 1.0
    if max_d < 1e-9:
        max_d = 1.0
    large_penalty = 10.0 * max_d

    # Threshold for 'burning' an edge: grows with round number
    # In early rounds, burn edges quickly; in later rounds, be more selective
    burn_threshold = 2.0 + 0.5 * np.log1p(round_num)

    # Mild penalty for edges below threshold
    mild_factor = 1.0 + 0.15 * np.log1p(round_num)

    for u, v in tour_edges:
        usage = float(U[u, v])
        d_uv = D[u, v]

        if usage >= burn_threshold:
            # Burn this edge: make it very expensive
            D[u, v] = large_penalty
            D[v, u] = large_penalty
        else:
            # Mild penalty: encourage avoiding this edge
            # Scale penalty by relative length (longer edges get more penalty)
            rel_len = d_uv / avg_len if avg_len > 0 else 1.0
            rel_len = np.clip(rel_len, 0.5, 3.0)
            
            # Usage-scaled mild penalty
            usage_factor = 1.0 + 0.2 * (usage / max(burn_threshold, 1.0))
            new_d = d_uv * mild_factor * (0.8 + 0.2 * rel_len) * usage_factor
            D[u, v] = new_d
            D[v, u] = new_d

    # Ensure symmetry
    D = (D + D.T) * 0.5
    # Ensure finite
    D = np.nan_to_num(D, nan=0.0,
                      posinf=np.finfo(np.float64).max / 10,
                      neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(D, 0.0)

    return D
