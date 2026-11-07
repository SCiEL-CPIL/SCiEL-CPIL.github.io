import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    
    # Calculate a scale factor based on the typical edge length
    # Mask out the diagonal to get mean of actual edges
    mask = ~np.eye(n, dtype=bool)
    if np.any(mask):
        avg_len = np.mean(edge_distance[mask])
    else:
        avg_len = 1.0
        
    # Ensure avg_len is not zero to avoid division issues or zero penalty
    if avg_len < 1e-10:
        avg_len = 1.0
        
    # Create the modified distance matrix
    modified_dist = edge_distance.copy().astype(float)
    
    # Determine current round index from max usage count
    max_usage = np.max(edge_n_used) if edge_n_used.size > 0 else 0
    current_round = max_usage  # Approximate current round
    
    # Hybrid reward-penalty:
    # 1. Historical penalty: penalize edges that have been used many times
    #    Penalty grows with usage count
    # 2. Reward: edges with low usage get a small negative offset to encourage exploration
    # 3. Strong penalty on current tour edges to force diversification
    
    # Base historical penalty: proportional to usage count
    # Use a smooth function to avoid discontinuities
    usage = edge_n_used.astype(float)
    
    # Penalty for high-usage edges (used more than current_round/2)
    # Reward for low-usage edges (used less than current_round/4)
    threshold_high = max(current_round * 0.5, 1.0)
    threshold_low = max(current_round * 0.25, 0.5)
    
    # Smooth penalty: increases with usage above threshold
    high_mask = usage > threshold_high
    penalty_high = 0.03 * avg_len * np.where(high_mask, (usage - threshold_high) / (current_round + 1), 0.0)
    
    # Smooth reward: decreases with usage below threshold
    low_mask = usage < threshold_low
    reward_low = -0.01 * avg_len * np.where(low_mask, (threshold_low - usage) / (threshold_low + 1), 0.0)
    
    # Apply historical penalty and reward
    modified_dist += penalty_high + reward_low
    
    # Strong penalty on current tour edges to force diversification
    # Identify edges in the current tour
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        if u != v:
            modified_dist[u, v] += 0.15 * avg_len
            modified_dist[v, u] += 0.15 * avg_len
    
    # Ensure all distances are positive (TSP requires positive edge lengths)
    # Find the minimum off-diagonal value
    off_diag_mask = ~np.eye(n, dtype=bool)
    min_off_diag = np.min(modified_dist[off_diag_mask]) if np.any(off_diag_mask) else 0.0
    
    if min_off_diag < 1e-6:
        # Shift all off-diagonal elements to be at least 1e-6
        shift = 1e-6 - min_off_diag
        modified_dist[off_diag_mask] += shift
    
    # Ensure diagonal is 0
    np.fill_diagonal(modified_dist, 0.0)
    
    # Ensure symmetry
    modified_dist = 0.5 * (modified_dist + modified_dist.T)
    
    # Ensure finiteness
    modified_dist = np.nan_to_num(modified_dist, nan=0.0, posinf=avg_len, neginf=1e-6)
    
    return modified_dist
