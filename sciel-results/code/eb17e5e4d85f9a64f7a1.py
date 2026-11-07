import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Static Additive Penalty for Guided Local Search.
    
    Strategy:
    Apply a fixed, strong additive penalty to all edges that have been used 
    in any previous local optimum tour. This creates a "tabu" effect where 
    previously used edges become prohibitively expensive, forcing the 2-opt 
    heuristic to find alternative paths and escape the local basin of attraction 
    more effectively than a gradual adaptive penalty.
    """
    # Safety check for empty or invalid inputs
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    # Check if any edges have been used
    if np.max(edge_n_used) == 0:
        return edge_distance.copy()
    
    # Fixed penalty value.
    # Chosen to be significantly larger than the typical difference between 
    # alternative edges in a 2-opt move, ensuring that used edges are 
    # strongly discouraged.
    # For 100 points in unit square, avg edge length is ~0.01-0.05, 
    # but long edges can be up to ~1.4.
    # A penalty of 0.5 is substantial but not infinite.
    penalty_value = 0.5
    
    # Create a mask for edges that have been used at least once
    mask = edge_n_used > 0
    
    # Apply the additive penalty
    # We create a copy to avoid modifying the input
    perturbed_distance = edge_distance.copy()
    perturbed_distance[mask] += penalty_value
    
    return perturbed_distance
