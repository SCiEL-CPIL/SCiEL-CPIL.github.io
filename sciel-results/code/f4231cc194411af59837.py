"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Compute perturbed edge distances for guided local search.
    
    Strategy:
    1. Penalize edges in the current tour proportional to their usage count.
    2. Apply a repulsive force field: penalize edges that are geometrically close
       to tour edges (using cross-distances) to create an exclusion zone.
    3. Scale penalties relative to the median edge distance for robustness.
    4. Use smooth, bounded penalties to maintain a well-conditioned matrix for 2-opt.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    n = edge_distance.shape[0]
    
    # Compute scale factor: median of non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    # Ensure dist_scale is positive
    dist_scale = max(dist_scale, 1e-10)
    
    # Initialize penalty matrix
    penalty = np.zeros_like(edge_distance)
    
    # Extract tour edges
    tour = list(local_opt_tour)
    if len(tour) != n:
        # Fallback: just return original
        return edge_distance.copy()
    
    # Build list of tour edges (i, j) where i < j
    tour_edges = []
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        if i > j:
            i, j = j, i
        tour_edges.append((i, j))
    
    # Part 1: Penalize tour edges based on usage count
    # Use a smooth, bounded penalty: penalty = alpha * usage / (1 + usage)
    # This avoids unbounded penalties while still scaling with usage
    alpha = 0.5  # Base penalty strength
    
    max_usage = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_usage > 0:
        # Normalize usage: usage / (1 + usage) gives values in [0, 1)
        for (i, j) in tour_edges:
            usage = edge_n_used[i, j]
            if usage > 0:
                # Smooth bounded penalty
                penalty[i, j] = alpha * (usage / (1.0 + usage))
                penalty[j, i] = penalty[i, j]
    
    # Part 2: Repulsive force field - penalize edges close to tour edges
    # For each edge (a, b) not in the tour, compute its "closeness" to tour edges
    # using the minimum cross-distance.
    
    # Precompute tour edge set for O(1) lookup
    tour_edge_set = set(tour_edges)
    
    # For efficiency, we'll compute the repulsive penalty using a grid-based approach
    # or direct computation for small n. Since n=100, direct is fine.
    
    # For each non-tour edge, find the minimum distance to any tour edge
    # Using a soft-min approximation for smoothness
    
    beta = 0.3  # Strength of repulsive force
    gamma = 0.1  # Width parameter for the repulsive kernel
    
    # We'll compute this efficiently by iterating over all edges
    # For each edge (i, j), compute min over tour edges (a, b) of some distance metric
    
    # Distance between two edges (i,j) and (a,b): use the minimum of the four cross-distances
    # d(i,a), d(i,b), d(j,a), d(j,b)
    
    # For efficiency, precompute the tour edge endpoints
    tour_edges_arr = np.array(tour_edges)  # shape (n, 2)
    
    # For each non-tour edge, compute the repulsive penalty
    # We'll do this vectorized where possible
    
    # Create a mask for non-tour edges
    is_tour_edge = np.zeros((n, n), dtype=bool)
    for (i, j) in tour_edges:
        is_tour_edge[i, j] = True
        is_tour_edge[j, i] = True
    
    # For repulsive force, we consider all edges (including diagonal, but we'll mask later)
    # Compute cross-distances between each edge and each tour edge
    
    # This is O(n^2 * n) = O(n^3) which for n=100 is 1e6 operations - fine
    
    # For each edge (i, j), compute min cross-distance to any tour edge
    # min over (a,b) in tour_edges of min(d(i,a), d(i,b), d(j,a), d(j,b))
    
    # We'll compute this row by row for memory efficiency
    repulsive_penalty = np.zeros((n, n))
    
    # Precompute tour edge list as arrays
    a_arr = tour_edges_arr[:, 0]
    b_arr = tour_edges_arr[:, 1]
    
    # For each row i, compute the repulsive penalty for edges (i, j) for all j
    for i in range(n):
        # Cross-distances from i to all tour edge endpoints
        dist_i_to_a = edge_distance[i, a_arr]  # shape (n,)
        dist_i_to_b = edge_distance[i, b_arr]  # shape (n,)
        
        # For each j, we need min over tour edges of min(dist(i,a), dist(i,b), dist(j,a), dist(j,b))
        # We'll compute this for all j at once
        
        # dist_j_to_a: shape (n, n) - j x tour_edge_index
        # This is edge_distance[:, a_arr] but we need it for each j
        # Actually, let's think differently.
        
        # For a fixed i and varying j:
        # min over k (tour edge index) of min(d(i,a_k), d(i,b_k), d(j,a_k), d(j,b_k))
        
        # We can compute this as:
        # min_k min(d(i,a_k), d(i,b_k), d(j,a_k), d(j,b_k))
        # = min( min_k min(d(i,a_k), d(i,b_k)), min_k min(d(j,a_k), d(j,b_k)), 
        #          min_k min(d(i,a_k), d(j,b_k)), min_k min(d(i,b_k), d(j,a_k)) )
        # This doesn't factor nicely.
        
        # Let's just do it directly for each j, but vectorize over j
        # For each j, compute:
        # cross_dist[j, k] = min(edge_distance[i, a_k], edge_distance[i, b_k], 
        #                         edge_distance[j, a_k], edge_distance[j, b_k])
        
        # edge_distance[i, a_arr] is shape (n,) - same for all j
        # edge_distance[i, b_arr] is shape (n,) - same for all j
        # edge_distance[:, a_arr] is shape (n, n) - j x k
        # edge_distance[:, b_arr] is shape (n, n) - j x k
        
        dist_j_to_a = edge_distance[:, a_arr]  # shape (n, n)
        dist_j_to_b = edge_distance[:, b_arr]  # shape (n, n)
        
        # For each j, k: cross_dist = min(d_i_a[k], d_i_b[k], d_j_a[j,k], d_j_b[j,k])
        # We want min over k of this
        
        # Compute element-wise min
        cross_dist = np.minimum(np.minimum(dist_i_to_a[np.newaxis, :], 
                                           dist_i_to_b[np.newaxis, :]),
                                np.minimum(dist_j_to_a, dist_j_to_b))
        # shape (n, n) where row j, col k is the cross-distance for edge (i,j) and tour edge k
        
        # Now take min over k (axis=1)
        min_cross_dist = np.min(cross_dist, axis=1)  # shape (n,)
        
        # Apply repulsive kernel: penalty = beta * exp(-min_cross_dist / (gamma * dist_scale))
        # This gives high penalty for edges very close to tour edges, decaying smoothly
        repulsive_row = beta * np.exp(-min_cross_dist / (gamma * dist_scale))
        
        # Don't penalize the tour edges themselves (they're already handled in Part 1)
        repulsive_row[is_tour_edge[i, :]] = 0
        repulsive_row[i] = 0  # No self-penalty
        
        repulsive_penalty[i, :] = repulsive_row
    
    # Check time
    if time.time() > deadline:
        # Return what we have so far
        result = edge_distance + penalty + repulsive_penalty
        return result
    
    # Combine penalties
    total_penalty = penalty + repulsive_penalty
    
    # Make symmetric
    total_penalty = (total_penalty + total_penalty.T) / 2.0
    
    # Ensure no negative distances
    result = edge_distance + total_penalty
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, edge_distance)
    
    # Ensure non-negative
    result = np.maximum(result, 0)
    
    # Ensure diagonal is 0
    np.fill_diagonal(result, 0)
    
    return result
# EVOLVE-BLOCK-END
