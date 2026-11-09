import numpy as np

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # Compute k-Nearest Neighbors for each node
    # k=5 is a reasonable choice for n=50
    k = 5
    
    # Mask for edges that are in the k-NN graph (symmetric)
    knn_mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        # Get distances from i to all other nodes
        dists = distance_matrix[i].copy()
        dists[i] = np.inf # Ignore self
        
        # Find indices of k smallest distances
        indices = np.argsort(dists)
        nearest_neighbors = indices[:k]
        
        # Mark these neighbors in the mask
        # We mark both directions to ensure symmetry in attractiveness
        for j in nearest_neighbors:
            knn_mask[i, j] = True
            knn_mask[j, i] = True

    # Construct final heuristic
    # If the edge is in the k-NN graph, keep the base heuristic
    # If not, reduce it significantly to discourage long edges
    heuristic = base_heuristic.copy()
    
    # Reduce non-kNN edges
    reduction_factor = 0.01
    heuristic[~knn_mask] *= reduction_factor
    
    return heuristic
