import numpy as np

def heuristics(distance_matrix):
    """
    Computes an edge-attractiveness matrix for ACO TSP.
    Uses inverse squared distance with nearest-neighbor boosts to favor local, non-crossing edges.
    """
    n = distance_matrix.shape[0]
    
    # 1. Base heuristic: Inverse squared distance
    # This strongly penalizes long edges, which is beneficial for TSP
    # as optimal tours are local (planar).
    # Add small epsilon to avoid division by zero (though diagonal is 0, we don't use it)
    dist_sq = distance_matrix ** 2
    eta = 1.0 / (dist_sq + 1e-9)
    
    # Set diagonal to 0 as self-loops are not allowed/needed
    np.fill_diagonal(eta, 0.0)
    
    # 2. Nearest Neighbor Boost
    # Identify k-nearest neighbors for each node
    k = 4
    # For each row i, find indices of k smallest distances
    # distance_matrix[i, :] contains distances from i to all j
    # We need to be careful with the diagonal (0 distance)
    # Create a copy to ignore self-loops in sorting
    dm_copy = distance_matrix.copy()
    np.fill_diagonal(dm_copy, np.inf)
    
    # Get indices of k smallest distances for each node
    # argpartition is faster than full sort
    nn_indices = np.argpartition(dm_copy, kth=k-1, axis=1)[:, :k]
    
    # Create a boolean mask for nearest neighbors
    # nn_mask[i, j] is True if j is in the k-NN of i
    nn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        nn_mask[i, nn_indices[i]] = True
        
    # Symmetrize the neighbor relationship for the boost
    # If j is a neighbor of i, or i is a neighbor of j, we consider them connected
    # Mutual neighbors (both ways) get a stronger boost
    
    # Boost 1: One-way neighbor
    # If (i, j) is a NN link in either direction
    neighbor_link = nn_mask | nn_mask.T
    
    # Boost 2: Mutual neighbors
    mutual_link = nn_mask & nn_mask.T
    
    # Apply boosts
    # Base eta is already computed
    # Boost mutual neighbors by factor 2.0
    eta[mutual_link] *= 2.0
    # Boost one-way neighbors (that are not mutual) by factor 1.5
    # We need to exclude mutual ones from this step to avoid double boosting or just apply sequentially
    one_way_only = neighbor_link & ~mutual_link
    eta[one_way_only] *= 1.5
    
    # Ensure no negative values (shouldn't happen with positive distances and boosts)
    # And handle any non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
