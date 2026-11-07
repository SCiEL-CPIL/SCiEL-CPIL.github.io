"""Guided local search penalty for TSP."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: penalize edges in the current local
    optimum tour, weighted by how frequently they have been used.
    
    The penalty is applied multiplicatively: modified_dist = dist * (1 + alpha * w)
    where w is a normalized usage weight for edges in the current tour.
    
    Key design choices:
    - Only penalize edges that are in the current local-optimum tour (classic GLS).
    - Scale penalty by accumulated usage so repeatedly-used edges get stronger penalties.
    - Use a smooth saturating function to avoid extreme penalties.
    - Keep penalties moderate to allow 2-opt to still find good tours.
    """
    n = edge_distance.shape[0]
    modified = edge_distance.copy()
    
    # Identify edges in the current tour
    # local_opt_tour is a sequence of node indices forming a closed tour
    # Edge (i, j) is in the tour if tour[k] and tour[k+1] are consecutive in the tour
    # (with wrap-around for the last edge)
    
    # Extract tour edges
    tour = np.array(local_opt_tour, dtype=np.int64)
    m = len(tour)
    if m < 3:
        return modified
    
    # For each consecutive pair in the tour (including last->first), get the usage count
    # and apply a penalty
    for k in range(m):
        i = tour[k]
        j = tour[(k + 1) % m]
        
        # Get the usage count for this edge
        usage = edge_n_used[i, j]
        
        # Smooth saturating penalty weight:
        # w = usage / (1 + usage)  -> in [0, 1), approaches 1 for large usage
        # This gives: low usage -> small penalty, high usage -> penalty approaches alpha
        w = usage / (1.0 + usage)
        
        # Base penalty strength: we want the penalty to be meaningful but not overwhelming
        # alpha controls the maximum multiplicative factor increase
        # For a 100-point TSP, typical edge lengths are ~0.05-0.5
        # A penalty of 0.3-0.5 on frequently used edges should encourage escape
        alpha = 0.5
        
        # Apply penalty to both directions (symmetric)
        penalty_factor = 1.0 + alpha * w
        modified[i, j] *= penalty_factor
        modified[j, i] *= penalty_factor
    
    # Ensure the diagonal remains zero (distance from node to itself)
    np.fill_diagonal(modified, 0.0)
    
    # Ensure all values are finite and non-negative
    modified = np.maximum(modified, 0.0)
    modified = np.where(np.isfinite(modified), modified, edge_distance)
    
    return modified
# EVOLVE-BLOCK-END
