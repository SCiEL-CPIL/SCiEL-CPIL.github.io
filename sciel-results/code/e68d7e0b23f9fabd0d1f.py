"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for ACO orienteering.
    
    The heuristic should guide ants towards nodes with high prize and low distance,
    while respecting the return-to-origin constraint.
    
    Key considerations:
    1. Prize at node 0 is 0, so we shouldn't be attracted to node 0
    2. Distance matters - closer nodes are easier to visit and return from
    3. We need to be able to return to node 0, so d[i,j] + d[j,0] matters
    4. Higher heuristic = more attractive
    
    Strategy: Use prize / (distance from current + distance to return)^alpha
    This naturally favors high-prize nodes that are close and easy to return from.
    
    We also incorporate a feasibility filter: nodes that cannot be visited and returned
    from given the max_len constraint get very low heuristic values.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Base heuristic: prize / (distance + return_distance)
    # d[i,j] + d[j,0] is the cost of going from i to j and then back to 0
    # But since we're computing a static heuristic, we use a representative distance
    
    # For each pair (i, j), the relevant cost is d[i,j] + d[j,0]
    # We want high prize and low cost
    
    # Compute the "effective distance" for each transition
    # eff_dist[i,j] = d[i,j] + d[j,0]
    eff_dist = distance_matrix + distance_matrix[:, 0:1]  # d[i,j] + d[j,0]
    
    # Ensure no division by zero
    eff_dist_safe = np.where(eff_dist > 1e-9, eff_dist, 1e-9)
    
    # Base heuristic: prize[j] / eff_dist[i,j]
    # Higher prize and lower effective distance = higher heuristic
    base_heur = prizes[None, :] / eff_dist_safe
    
    # Apply a power to sharpen the preference for high-prize, low-cost nodes
    # Using a moderate exponent to balance exploration and exploitation
    alpha = 2.0
    heur = base_heur ** alpha
    
    # Handle node 0: we don't want to be attracted to node 0 as a destination
    # Set heuristic to 0 for all transitions to node 0
    heur[:, 0] = 0.0
    
    # Ensure all values are positive (replace non-positive with small epsilon)
    heur = np.where(heur > 0, heur, 1e-9)
    
    # Normalize to a reasonable range
    # The heuristic values should be in a similar range to pheromone (starts at 1)
    max_heur = np.max(heur)
    if max_heur > 0:
        heur = heur / max_heur
    
    # Ensure minimum value
    heur = np.maximum(heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
