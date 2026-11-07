"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP using a structural
    approach based on mutual rank proximity.
    
    Instead of simple inverse distance, we use the mutual rank of two nodes
    (the maximum of each node's rank of the other) to model local connectivity.
    Edges with low mutual rank (i.e., both nodes consider each other as close)
    are boosted, while long-range edges are penalized. This is more robust to
    density variations and better captures the local structure needed for a
    valid TSP tour.
    """
    n = distance_matrix.shape[0]
    D = distance_matrix.astype(np.float64)
    
    # Compute rank matrix: rank[i, j] = rank of j in i's sorted neighbors
    # rank[i, j] = number of nodes k (k != i) such that D[i, k] < D[i, j]
    # We can compute this efficiently using argsort
    rank = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Get indices sorted by distance from i
        sorted_indices = np.argsort(D[i])
        # The rank of each node j from i's perspective is its position in sorted_indices
        # But we want rank[i, j] = position of j in sorted_indices (0-indexed)
        # Create a rank array where rank[i, sorted_indices[k]] = k
        rank[i, sorted_indices] = np.arange(n)
    
    # Mutual rank: max of the two ranks
    mutual_rank = np.maximum(rank, rank.T)
    
    # Convert mutual rank to an attractiveness score
    # Lower mutual rank = more attractive
    # Use inverse of (mutual_rank + 1) to avoid division by zero
    # This gives a smooth decay: rank 0 -> 1.0, rank 1 -> 0.5, rank 2 -> 0.33, etc.
    mutual_attract = 1.0 / (mutual_rank + 1.0)
    
    # Also compute inverse distance as a base
    inv_dist = 1.0 / (D + 1e-9)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Combine: weight the mutual rank attractiveness with inverse distance
    # Use a geometric mean to balance both factors
    # heuristic = sqrt(inv_dist * mutual_attract)
    # This ensures that both local proximity (distance) and mutual connectivity (rank) matter
    heuristic = np.sqrt(inv_dist * mutual_attract)
    
    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Clamp to positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
