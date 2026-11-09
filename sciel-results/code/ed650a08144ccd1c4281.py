"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    Returns a modified distance matrix that penalizes frequently used edges,
    encouraging the subsequent 2-opt to explore different tour structures.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Get the edges in the current local optimal tour
    # The tour is a permutation of nodes; edges are (tour[i], tour[i+1]) and (tour[-1], tour[0])
    tour = np.asarray(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Create a set of edges in the current tour for quick lookup
    # We'll work with the edge_n_used matrix directly
    
    # Compute a global usage-based penalty
    # edge_n_used is symmetric count matrix
    # We want to penalize edges that have been used frequently
    
    # Normalize edge_n_used to get relative usage
    max_usage = edge_n_used.max()
    if max_usage == 0:
        return edge_distance.copy()
    
    # Strategy: combine local (current tour) and global (historical) penalties
    # Local penalty: penalize edges in the current tour more strongly
    # Global penalty: penalize all frequently used edges
    
    # Create penalty matrix
    penalty = np.zeros_like(edge_distance)
    
    # Global penalty: proportional to usage count
    # Use a smooth function to avoid discontinuities
    # penalty_global = alpha * (edge_n_used / (1 + edge_n_used))
    # This gives a penalty that saturates at alpha for very high usage
    
    alpha_global = 0.3
    penalty_global = alpha_global * (edge_n_used / (1.0 + edge_n_used))
    
    # Local penalty: extra penalty for edges in the current tour
    # This ensures the current tour's edges are strongly discouraged
    alpha_local = 0.5
    penalty_local = np.zeros_like(edge_distance)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        penalty_local[u, v] = alpha_local
        penalty_local[v, u] = alpha_local
    
    # Combine penalties
    total_penalty = penalty_global + penalty_local
    
    # Apply penalty to the distance matrix
    # Multiplicative form: d' = d * (1 + penalty)
    # This ensures the penalty scales with the original distance
    modified_distance = edge_distance * (1.0 + total_penalty)
    
    # Ensure symmetry
    modified_distance = (modified_distance + modified_distance.T) / 2.0
    
    # Ensure finite values
    modified_distance = np.nan_to_num(modified_distance, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Safety: check elapsed time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Return a simple perturbation if we're running out of time
        return edge_distance * (1.0 + 0.1 * (edge_n_used / (1.0 + edge_n_used)))
    
    return modified_distance
# EVOLVE-BLOCK-END
