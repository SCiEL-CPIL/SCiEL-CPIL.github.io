"""Guided local search penalty rule for the TSP with local tour-specific penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - Apply a strong, localized additive penalty ONLY to the edges currently 
      in the local_opt_tour.
    - The penalty magnitude is proportional to the square root of the usage count 
      of that edge, ensuring that frequently used edges in the current tour are 
      heavily penalized, forcing the local search to escape the current basin.
    - Edges not in the current tour retain their true distances, preserving the 
      geometric structure of the problem and allowing the search to move to 
      nearby, unvisited regions of the solution space.
    - This approach avoids the global distortion of the distance matrix that 
      can mislead the local search, focusing the perturbation exactly where it 
      is needed to break the local optimum.
    """
    # Check deadline: if we are too close to the limit, return a safe, 
    # minimally perturbed matrix to avoid timeout.
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    D = edge_distance.copy()

    # Calculate a scale factor based on the average edge length of the current tour.
    # This ensures the penalty is proportional to the problem's geometry.
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # If the tour length is zero or invalid, use a default scale
    if not np.isfinite(avg_edge_len) or avg_edge_len <= 0:
        avg_edge_len = 1.0

    # Define the penalty coefficient.
    # We want a strong penalty to break the local optimum.
    # alpha = 1.0 * avg_edge_len means an edge used 1 time gets a penalty 
    # of 1.0 * avg_edge_len, which is substantial and likely to force a change.
    alpha = 1.0 * avg_edge_len

    # Apply the penalty ONLY to the edges in the current tour.
    # For each edge (u, v) in the tour, add alpha * sqrt(edge_n_used[u, v]) to D[u, v] and D[v, u].
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Usage count for this edge
        count = edge_n_used[u, v]
        if count > 0:
            # Penalty proportional to sqrt(count) to balance strong repulsion with stability
            penalty = alpha * np.sqrt(count)
            D[u, v] += penalty
            D[v, u] += penalty

    # Ensure symmetry and finiteness
    # The addition preserves symmetry if input was symmetric and penalty is symmetric.
    # We added symmetric penalties, so D should remain symmetric.
    
    if not np.all(np.isfinite(D)):
        # Fallback to true distances if something went wrong
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
