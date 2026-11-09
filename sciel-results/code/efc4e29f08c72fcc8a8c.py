"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Combines:
    1. A global soft additive penalty proportional to edge usage frequency,
       which gently discourages frequently-used edges across the whole matrix.
    2. A stronger targeted multiplicative penalty on edges in the current
       local-optimal tour, scaled by their usage count, to specifically break
       the current solution.
    
    This dual approach ensures the search can escape local minima by both
    weakening globally popular edges and strongly perturbing the current tour.
    """
    t0 = time.time()
    
    n = edge_distance.shape[0]
    # Make a copy so we don't mutate the input
    penalized = edge_distance.copy()
    
    # Extract edges from the current tour
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    
    if m < 2:
        return penalized
    
    # Get usage statistics
    max_used = edge_n_used.max()
    if max_used <= 0:
        max_used = 1.0
    
    # --- Component 1: Global soft penalty ---
    # Add a small penalty to all edges proportional to their usage count.
    # This gently discourages edges that have been used frequently.
    # Use a small coefficient to avoid overwhelming the true distances.
    global_coeff = 0.05
    penalized += global_coeff * edge_n_used
    
    # --- Component 2: Targeted penalty on current tour edges ---
    # For each edge in the current tour, apply a stronger multiplicative penalty
    # that depends on how many times that edge has been used.
    # This specifically targets the structure of the current local optimum.
    
    # Precompute usage for tour edges for efficiency
    tour_edges = []
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        tour_edges.append((u, v))
    
    # Apply targeted penalties
    # Base multiplier + usage-scaled increase
    base_mult = 1.3
    usage_scale = 0.4
    cap_mult = 4.0
    
    for u, v in tour_edges:
        used = edge_n_used[u, v]
        # Penalty factor: grows with usage, capped to avoid extreme values
        penalty_factor = base_mult + usage_scale * (used / max_used)
        penalty_factor = min(penalty_factor, cap_mult)
        penalized[u, v] *= penalty_factor
        penalized[v, u] *= penalty_factor
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Time guard: if we're taking too long, return anyway
    # This function should take microseconds to milliseconds for n=100
    
    return penalized
# EVOLVE-BLOCK-END
