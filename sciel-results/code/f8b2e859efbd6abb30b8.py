"""Guided local search penalty rule for TSP with sparse tour-edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Sparse GLS penalty: add a constant penalty to edges in the current tour,
    scaled by average edge length. This forces the local search to escape
    the current basin by modifying the cost landscape only where stuck.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Create a copy of the true distance matrix to avoid mutating input
    penalized_dist = edge_distance.copy()

    # Extract edges of the current local optimum tour
    tour = np.asarray(local_opt_tour)
    u = tour
    v = np.roll(tour, -1)

    # Get true distances for these tour edges
    current_edge_lengths = edge_distance[u, v]
    avg_edge_len = np.mean(current_edge_lengths)

    # Fallback for degenerate cases
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1

    # Add a targeted penalty to the current tour edges to force diversification.
    # Using a fraction of the average edge length makes the penalty scale-invariant.
    # A coefficient of 0.5 means each current tour edge gets a penalty equal to
    # half the average edge length, making the current tour noticeably more expensive.
    penalty = 0.5 * avg_edge_len
    penalized_dist[u, v] += penalty
    penalized_dist[v, u] += penalty

    # Ensure the diagonal is 0 to avoid self-loops
    np.fill_diagonal(penalized_dist, 0.0)

    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0

    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)

    return penalized_dist
# EVOLVE-BLOCK-END
