import numpy as np
import time

# Global start time to track wall-clock limit
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Strategy:
    1. Check time limit; if exceeded, return original distances to avoid timeout.
    2. Calculate a dynamic penalty factor based on the average edge length of the current tour.
       This ensures the penalty is significant relative to the problem scale.
    3. Apply an additive penalty to edges that are part of the current local_opt_tour.
       The penalty is proportional to the number of times that specific edge has been used (edge_n_used).
       This creates a "ratchet" effect: frequently used edges in the local optimum become very expensive,
       forcing the subsequent 2-opt to break them.
    4. Ensure the returned matrix is finite and symmetric.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    
    # Ensure inputs are numpy arrays for safety
    edge_distance = np.asarray(edge_distance, dtype=float)
    edge_n_used = np.asarray(edge_n_used, dtype=float)
    local_opt_tour = np.asarray(local_opt_tour, dtype=int)
    
    # Copy the distance matrix to avoid modifying the original
    perturbed_distance = edge_distance.copy()
    
    if n <= 1:
        return perturbed_distance
        
    # Calculate the average edge length of the current tour to set a scale-invariant penalty base
    # local_opt_tour is a sequence of node indices
    tour_len = len(local_opt_tour)
    if tour_len < 2:
        return perturbed_distance
        
    # Compute sum of distances for edges in the current tour
    tour_edge_indices = []
    for i in range(tour_len - 1):
        u = local_opt_tour[i]
        v = local_opt_tour[i+1]
        tour_edge_indices.append((u, v))
    # Close the tour
    u = local_opt_tour[-1]
    v = local_opt_tour[0]
    tour_edge_indices.append((u, v))
    
    total_tour_dist = 0.0
    for u, v in tour_edge_indices:
        total_tour_dist += edge_distance[u, v]
        
    avg_edge_len = total_tour_dist / len(tour_edge_indices) if len(tour_edge_indices) > 0 else 1.0
    
    # If avg_edge_len is 0 or very small, use a small default to avoid division issues or zero penalty
    if avg_edge_len < 1e-9:
        avg_edge_len = 1.0
        
    # Dynamic penalty factor: 
    # We want the penalty to be significant enough to break the local optimum.
    # A common heuristic in GLS is alpha * (edge_length) or similar.
    # Here we use a factor that scales with the average edge length.
    # alpha = 1.5 means we add 1.5 * usage_count * avg_edge_len to the cost.
    # This makes used edges in the current tour significantly more expensive.
    alpha = 1.5
    
    # Apply penalty to edges in the current tour
    for u, v in tour_edge_indices:
        # Only penalize if the edge is actually in the matrix (u, v)
        if u < n and v < n:
            usage_count = edge_n_used[u, v]
            # Additive penalty proportional to usage and scale
            penalty = alpha * usage_count * avg_edge_len
            perturbed_distance[u, v] += penalty
            perturbed_distance[v, u] += penalty
            
    # Ensure symmetry and finiteness
    # Symmetry check
    perturbed_distance = (perturbed_distance + perturbed_distance.T) / 2.0
    
    # Handle any non-finite values (shouldn't happen with addition, but safe)
    if not np.all(np.isfinite(perturbed_distance)):
        perturbed_distance = np.nan_to_num(perturbed_distance, nan=avg_edge_len, posinf=avg_edge_len*100, neginf=0)
        
    return perturbed_distance
