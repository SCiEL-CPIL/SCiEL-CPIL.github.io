import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    if n == 0:
        return np.array([])
    
    # Ensure coordinates are float array
    coords = np.array(coordinates, dtype=float)
    dist_mat = np.array(distance_matrix, dtype=float)
    
    # Compute distance from depot (node 0) to all other nodes
    depot = coords[0]
    dist_to_depot = np.linalg.norm(coords - depot, axis=1)
    
    # Calculate savings for each pair (i, j)
    # Savings s_ij = d_0i + d_0j - d_ij
    # We vectorize this calculation
    d0 = dist_to_depot[:, np.newaxis]
    d0_t = dist_to_depot[np.newaxis, :]
    savings = d0 + d0_t - dist_mat
    
    # The savings heuristic is most effective for edges that connect nodes 
    # which would otherwise require separate trips to the depot.
    # We want to boost edges with high positive savings.
    # If savings <= 0, the edge is not beneficial in terms of savings, 
    # so we rely primarily on distance.
    
    # Normalize savings to [0, 1] range for scaling
    max_savings = np.max(savings)
    if max_savings > 0:
        norm_savings = savings / max_savings
    else:
        norm_savings = np.zeros_like(savings)
    
    # Clip savings to 0 so we don't penalize edges with negative savings too harshly
    # in the multiplicative term, but we still want distance to matter.
    # Heuristic: H_ij = (1/d_ij) * (1 + alpha * norm_savings)
    # If savings are negative, norm_savings is negative. 
    # We should probably floor the savings contribution to 0 to avoid 
    # explicitly discouraging edges that are necessary for connectivity 
    # but don't offer "savings" in the Clarke-Wright sense (e.g., last mile).
    
    positive_savings = np.maximum(0, norm_savings)
    
    # Base heuristic is inverse distance
    base_heur = 1.0 / (dist_mat + 1e-9)
    
    # Boost based on savings
    # alpha controls the strength of the savings bias. 
    # A value of 1.0 means savings can double the attractiveness of an edge.
    alpha = 1.5 
    heuristic_matrix = base_heur * (1.0 + alpha * positive_savings)
    
    # Ensure no zeros or negatives (though base_heur is always positive)
    heuristic_matrix[~np.isfinite(heuristic_matrix)] = 1e-9
    heuristic_matrix[heuristic_matrix <= 0] = 1e-9
    
    return heuristic_matrix
