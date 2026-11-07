"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Implements a strong edge-exclusion strategy: edges in the current local-optimal
    tour are set to a very large finite value (effectively excluded) to force the
    subsequent 2-opt to remove them, guaranteeing a distinct neighborhood search step.
    Additionally, a soft penalty is applied to frequently used edges for diversity.
    
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
    
    # Get the set of edges in the current tour
    tour_edges = set()
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u != v:
            tour_edges.add((min(u, v), max(u, v)))
    
    # Penalty strategy:
    # 1. Set tour edges to a very large finite value to force their removal
    # 2. Apply a soft penalty based on edge_n_used for additional diversity
    
    # Determine a large but finite value for excluded edges
    finite_vals = edge_distance[np.isfinite(edge_distance)]
    if len(finite_vals) > 0:
        max_dist = np.max(finite_vals)
    else:
        max_dist = 1.0
    if max_dist <= 0:
        max_dist = 1.0
    
    # Use a very large value to effectively exclude tour edges from 2-opt consideration
    # This should be large enough that 2-opt will always prefer to remove these edges
    exclusion_value = 1e6 * max_dist
    
    # Apply exclusion to tour edges
    for (u, v) in tour_edges:
        modified[u, v] = exclusion_value
        modified[v, u] = exclusion_value
    
    # Additional soft penalty based on edge_n_used for non-tour edges
    # This adds diversity to the search by penalizing frequently used edges
    if edge_n_used is not None and edge_n_used.size > 0:
        n_used = np.asarray(edge_n_used, dtype=float)
        # Ensure n_used has the right shape
        if n_used.shape != modified.shape:
            n_used = np.zeros_like(modified)
        # Normalize edge_n_used to [0, 1]
        max_used = np.max(n_used)
        if max_used > 0:
            normalized_used = n_used / max_used
            # Apply a moderate penalty to frequently used edges (not in current tour)
            soft_penalty = 0.5 * max_dist * normalized_used
            # Don't apply soft penalty to tour edges (they already have exclusion)
            for (u, v) in tour_edges:
                soft_penalty[u, v] = 0.0
                soft_penalty[v, u] = 0.0
            modified += soft_penalty
    
    # Ensure all values are finite
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Ensure symmetry
    modified = (modified + modified.T) / 2.0
    
    return modified
# EVOLVE-BLOCK-END
