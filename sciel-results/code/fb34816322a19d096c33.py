"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Implements a distance-preserving edge exclusion strategy: edges in the current
    local-optimal tour are set to a very large finite value (effectively infinite)
    to force the subsequent 2-opt to remove them. This guarantees a distinct
    neighborhood search step by breaking the current cycle.
    
    Args:
        edge_distance: True Euclidean distance matrix (n x n).
        local_opt_tour: Current tour as a list/array of node indices (closed tour).
        edge_n_used: Symmetric count matrix of how many times each edge has been used.
    
    Returns:
        A modified distance matrix of the same shape with finite values.
    """
    n = edge_distance.shape[0]
    modified = edge_distance.copy().astype(float)
    
    # Ensure finite values
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Extract edges from the current tour
    # The tour is a closed tour: tour[0] -> tour[1] -> ... -> tour[-1] -> tour[0]
    tour = np.asarray(local_opt_tour)
    
    # Handle potential edge cases
    if len(tour) < 2:
        return modified
    
    # Define a large penalty value that is finite but effectively infinite for 2-opt
    # This ensures that any swap involving these edges is highly favorable in the
    # modified metric, forcing 2-opt to remove them.
    LARGE_PENALTY = 1e10
    
    # Apply the exclusion penalty to all edges in the current tour
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u != v:
            modified[u, v] = LARGE_PENALTY
            modified[v, u] = LARGE_PENALTY
    
    # Ensure all values are finite (they should be, but just in case)
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Ensure symmetry (should already be symmetric due to how we set them, but safe)
    modified = (modified + modified.T) / 2.0
    
    return modified
# EVOLVE-BLOCK-END
