"""Guided local search penalty for TSP – multiplicative tour-edge scaling."""
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Return a perturbed distance matrix that heavily penalizes edges present
    in the current local-optimal tour, leaving all other edges at their true
    distances.  This forces the subsequent 2-opt to sever specific tour
    connections without distorting the relative geometry elsewhere.

    Parameters
    ----------
    edge_distance : np.ndarray, shape (n, n)
        True Euclidean distance matrix.
    local_opt_tour : sequence of int, length n
        Current tour as a cyclic sequence of node indices.
    edge_n_used : np.ndarray, shape (n, n)
        Symmetric count matrix of how often each edge has been used.

    Returns
    -------
    np.ndarray, shape (n, n)
        Perturbed distance matrix (finite, same shape).
    """
    n = edge_distance.shape[0]
    result = edge_distance.copy()

    if n < 2:
        return result

    # Identify the set of edges in the current tour.
    # The tour is a cycle: tour[i] -> tour[(i+1) % n]
    tour = list(local_opt_tour)
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))

    # Build a boolean mask for tour edges (symmetric).
    # We want a high multiplicative penalty ONLY on tour edges.
    # Use a penalty that scales with usage: more-used edges get higher penalty.
    # Base penalty factor: 5.0 (strong enough to force 2-opt to break the edge)
    # Scale with normalized usage to emphasize frequently-used edges.
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0.0
    if max_used > 0:
        # Normalized usage in [0, 1]
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)

    # For each tour edge, apply a multiplicative penalty.
    # Penalty factor = base * (1 + alpha * normalized_usage)
    # This makes heavily-used tour edges even more expensive.
    base_penalty = 5.0
    alpha = 2.0  # additional scaling with usage

    for (u, v) in tour_edges:
        usage = norm_used[u, v]
        factor = base_penalty * (1.0 + alpha * usage)
        result[u, v] *= factor
        result[v, u] *= factor

    return result
