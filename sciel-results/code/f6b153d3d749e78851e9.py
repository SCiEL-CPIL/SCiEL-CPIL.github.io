"""Guided local search penalty rule for TSP using sparse targeted edge-breaking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply a sparse, targeted penalty to break the current local optimum.
    
    Strategy:
    1. Identify edges in the current tour.
    2. Select the 'most stubborn' edges: those with the highest usage count in edge_n_used.
       If counts are equal, prefer longer edges (higher true distance) as they represent
       larger structural components.
    3. Apply a large penalty to these specific edges to force 2-opt to remove them,
       thereby escaping the current basin of attraction.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Ensure we have a working copy
    modified_distance = edge_distance.copy()
    
    # Extract the edges of the current tour
    # The tour is a list of node indices. Edges are (tour[i], tour[(i+1)%n])
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Ensure u < v for consistent indexing if needed, but edge_distance is symmetric
        tour_edges.append((u, v))
        
    if not tour_edges:
        return modified_distance

    # Get usage counts and true distances for these specific edges
    # We want to penalize the edges that have been used the most in the history.
    # This is the core of Guided Local Search: penalize the solution's defining features.
    
    # Create a list of (usage_count, distance, u, v)
    edge_stats = []
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        dist = edge_distance[u, v]
        edge_stats.append((count, dist, u, v))
    
    # Sort by usage count (descending), then by distance (descending)
    # We want to break the edges that are most "familiar" to the search.
    # If usage is 0 for all, we break the longest ones to maximize potential improvement.
    edge_stats.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Determine how many edges to penalize.
    # Penalizing too many makes the perturbation too strong and random.
    # Penalizing too few might not escape the basin.
    # A common heuristic is to penalize a small fraction, e.g., 5-10% of the tour.
    # For n=100, let's penalize the top 5 edges.
    # This is a "hard" kick.
    
    num_to_penalize = max(1, n // 20)  # 5% of edges, at least 1
    
    # The penalty value must be large enough to override the local 2-opt optimality.
    # If we add a penalty P, 2-opt will try to remove that edge if the alternative
    # path is shorter than d_edge + P.
    # To *force* removal, P should be larger than the maximum possible 2-opt improvement
    # from removing that edge. A safe large value is relative to the average edge length.
    
    # Calculate average edge length for scaling
    avg_len = np.mean([edge_distance[u, v] for u, v in tour_edges])
    
    # A penalty of 5 * avg_len is very strong. It essentially says "this edge is 5x more expensive".
    # This will almost certainly cause 2-opt to cut it.
    penalty_value = 5.0 * avg_len
    
    # Apply the penalty to the selected edges
    for count, dist, u, v in edge_stats[:num_to_penalize]:
        # Only penalize if the edge is actually in the matrix (it should be)
        # and if we haven't already penalized it (u,v unique in tour)
        modified_distance[u, v] += penalty_value
        modified_distance[v, u] += penalty_value
        
    return modified_distance
# EVOLVE-BLOCK-END
