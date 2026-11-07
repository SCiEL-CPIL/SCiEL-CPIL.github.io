"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
# Deadline is 330s, we use 320s to be safe
_DEADLINE = 320.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Uses an additive penalty model based on:
    1. Saturating usage penalty: penalizes edges that have been used frequently 
       in previous iterations.
    2. Immediate tour penalty: penalizes edges currently in the local optimum 
       to force escape.
    
    The penalty is added to the true distances.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return original distances if we are out of time (should not happen for n=100)
        return edge_distance.copy()

    n = edge_distance.shape[0]
    
    # --- 1. Usage-based penalty ---
    # We use a saturating function: P(u) = alpha * u / (beta + u)
    # This ensures the penalty is bounded by alpha, preventing numerical issues
    # while still strongly penalizing frequently used edges.
    alpha = 0.8
    beta = 1.0
    
    # Ensure edge_n_used is float for division
    usage = edge_n_used.astype(np.float64)
    usage_penalty = alpha * usage / (beta + usage)
    
    # --- 2. Immediate penalty on current tour edges ---
    # Edges in the current local optimum should be penalized immediately
    # to guarantee that the next 2-opt run does not converge to the exact same tour.
    immediate_scale = 0.4
    
    # Create a mask for edges in the local_opt_tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
        
    # Combine penalties
    # The total penalty is the sum of the usage penalty and the immediate penalty
    # Note: If an edge is in the tour, it likely has usage > 0, so penalties stack.
    total_penalty = usage_penalty + immediate_scale * tour_mask
    
    # --- 3. Apply penalty additively ---
    # Additive penalties are standard for GLS and provide a clear cost increase.
    perturbed = edge_distance + total_penalty
    
    # Ensure symmetry (should already be symmetric, but just in case)
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is zero (distance from node to itself is 0)
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure all values are finite and non-negative
    # Distances should be positive, penalties are positive, so sum should be positive.
    # However, if there's any NaN/Inf, we replace with a large value or 0.
    # Replacing with 0 might break the triangle inequality, but 0 is safe for validity.
    # Replacing with a large value (like max distance) might be better to discourage use.
    # Let's just ensure finiteness.
    perturbed = np.where(np.isfinite(perturbed), perturbed, 0.0)
    
    # Ensure non-negative
    perturbed = np.maximum(perturbed, 0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
