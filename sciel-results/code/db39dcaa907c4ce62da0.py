import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Context-aware multiplicative penalty for Guided Local Search.
    
    Strategy:
    1. Identify edges in the current local optimum tour.
    2. For each edge in the tour, calculate a penalty factor based on:
       a. Its usage count (more used = more penalized).
       b. Its relative length (longer edges are penalized more heavily, 
          assuming they are more likely to be part of the suboptimal structure).
    3. Apply a multiplicative penalty to these specific edges in the distance matrix.
    
    This approach is "surgical": it only distorts the costs of edges currently 
    in the tour, forcing the subsequent 2-opt to find alternatives for the 
    most problematic (frequently used and long) edges, while leaving other 
    potential edges' costs undistorted. This is often more effective than 
    global additive penalties which can make the entire search space 
    inconsistent with the true geometry.
    """
    # Safety check
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    # Create a copy of the distance matrix to modify
    modified_distance = edge_distance.copy()
    
    # Get the edges in the current tour
    # local_opt_tour is a list of node indices, e.g., [0, 1, 2, ..., 99, 0]
    # We need to extract pairs (i, j)
    n = len(local_opt_tour)
    if n < 2:
        return modified_distance
    
    # Get max usage for normalization
    max_used = np.max(edge_n_used)
    if max_used == 0:
        return modified_distance
    
    # Get max edge length in the tour for normalization
    # We'll use the true distances from edge_distance
    tour_edges = []
    for i in range(n - 1):
        u = local_opt_tour[i]
        v = local_opt_tour[i + 1]
        tour_edges.append((u, v))
    # Add the closing edge
    tour_edges.append((local_opt_tour[-1], local_opt_tour[0]))
    
    # Calculate relative lengths to identify "long" edges
    # Normalizing by the average edge length in the tour or just using raw length
    # Since points are in unit square, avg edge length is ~0.3-0.5
    # We want to penalize edges that are significantly longer than average
    # Let's use a simple heuristic: penalty scales with (length / mean_length)
    
    # Compute mean length of tour edges
    lengths = []
    for u, v in tour_edges:
        lengths.append(edge_distance[u, v])
    mean_len = np.mean(lengths) if len(lengths) > 0 else 1.0
    
    # Base penalty factor
    # We want the penalty to be significant but not infinite.
    # A multiplicative factor of 1.0 to 2.0 is reasonable.
    # Let's say max penalty factor is 1.5 (50% increase in cost)
    
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        if count == 0:
            continue
        
        # Usage factor: linear scaling, normalized
        # If an edge is used max_used times, it gets the max usage penalty.
        usage_factor = count / max_used
        
        # Length factor: penalize long edges more
        # If an edge is twice the mean length, it gets double the length penalty component.
        # We clamp this to avoid extreme values.
        length_ratio = edge_distance[u, v] / mean_len if mean_len > 0 else 1.0
        length_factor = min(length_ratio, 2.0) # Cap at 2x
        
        # Combine factors
        # penalty_multiplier = 1.0 + base_penalty * usage_factor * length_factor
        # Let's choose base_penalty such that the max case (usage=1, length=2) gives a factor of ~1.5
        # 1.0 + base * 1 * 2 = 1.5 => base = 0.25
        # But we want it to scale with usage.
        # Let's try: multiplier = 1.0 + 0.5 * usage_factor * length_factor
        # Max multiplier = 1.0 + 0.5 * 1.0 * 2.0 = 2.0 (100% increase)
        
        multiplier = 1.0 + 0.5 * usage_factor * length_factor
        
        # Apply to both directions (symmetric)
        modified_distance[u, v] *= multiplier
        modified_distance[v, u] *= multiplier
        
    return modified_distance
