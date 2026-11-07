"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Apply a high multiplicative penalty factor ONLY to edges in the current
       local-optimal tour, leaving all other edges at their true distances.
       This forces 2-opt to sever specific tour connections without distorting
       the relative geometry of the remaining search space.
    2. The penalty factor is modulated by cumulative edge usage: edges that have
       been used many times receive a slightly stronger penalty to encourage
       exploration of less-frequently-used connections.
    
    All operations are vectorized and fast to stay within time limits.
    """
    n = edge_distance.shape[0]
    
    # Deadline guard: track start time
    t_start = time.time()
    
    # --- Step 1: Build tour mask for current local-opt tour (vectorized) ---
    tour_arr = np.asarray(local_opt_tour, dtype=np.int64)
    a = tour_arr[:-1]
    b = tour_arr[1:]
    # Also need the last edge connecting last city to first city
    a_last = tour_arr[-1]
    b_last = tour_arr[0]
    
    tour_mask = np.zeros((n, n), dtype=np.float64)
    tour_mask[a, b] = 1.0
    tour_mask[b, a] = 1.0
    tour_mask[a_last, b_last] = 1.0
    tour_mask[b_last, a_last] = 1.0
    
    # --- Step 2: Compute usage-based modulation for tour edges ---
    # Edges with higher cumulative usage get a slightly stronger penalty.
    # Use a smooth saturation: usage / (1 + usage) maps [0, inf) -> [0, 1)
    usage_sat = edge_n_used / (1.0 + edge_n_used)
    
    # Base penalty factor for current tour edges
    base_factor = 2.5
    
    # Modulate: edges used more often get up to 30% stronger penalty
    # usage_sat on tour edges is in [0, 1), so modulation is in [1, 1.3]
    modulation = 1.0 + 0.3 * usage_sat
    
    # Final penalty factor: only applies to tour edges
    penalty_factor = base_factor * modulation
    penalty_factor = penalty_factor * tour_mask  # zero everywhere except tour edges
    
    # --- Step 3: Apply multiplicative scaling ---
    # For tour edges: distance * base_factor * modulation
    # For non-tour edges: distance * 1.0 (unchanged)
    result = edge_distance * (1.0 + penalty_factor)
    
    # --- Step 4: Ensure finite values ---
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # --- Step 5: Deadline guard ---
    t_elapsed = time.time() - t_start
    # This function is called 20 times per instance, 3 instances total = 60 calls.
    # Each call should be fast. For n=100, this is negligible. No early exit needed.
    
    return result
# EVOLVE-BLOCK-END
