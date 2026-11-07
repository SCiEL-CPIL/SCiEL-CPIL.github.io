"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic based on local density of value.
    
    Strategy:
    1. For each node j, calculate a 'sparsity' score based on the average distance
       to its k nearest neighbors.
    2. The heuristic h[i, j] is proportional to the prize of node j divided by its
       sparsity score. This rewards moving to high-prize nodes that are clustered
       with other nodes (low sparsity), facilitating efficient collection.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    # Choose k nearest neighbors for density calculation
    k = min(5, n - 1)
    
    # Calculate sparsity for each node j:
    # Sort distances from j to all other nodes, take the first k, and average them.
    # This gives a measure of how "isolated" node j is.
    # Lower sparsity means node j is in a dense cluster.
    
    # distance_matrix is (n, n). For each row j, we want the k smallest values excluding self (dist=0).
    # We can use np.partition or np.sort. np.partition is faster.
    
    # Add a small epsilon to avoid division by zero if distances are 0 (though unlikely in random points)
    eps = 1e-9
    
    # We need the k-th smallest distance for each row, and the sum of the k smallest.
    # np.partition(arr, k) puts the k-th smallest element at index k, with all smaller elements
    # to its left (order not guaranteed among them).
    
    # Mask out the diagonal (self-distance) by setting it to infinity before partitioning
    dm_work = distance_matrix.copy()
    np.fill_diagonal(dm_work, np.inf)
    
    # Get the k smallest distances for each node
    # If n-1 < k, k is already capped.
    if k > 0:
        # partition along axis 1
        # We want the first k elements.
        # Note: np.partition requires k < n. Here we have n elements per row, one is inf.
        # So valid elements are n-1.
        # We want the k smallest of the n-1 valid elements.
        # So we partition at index k.
        # The elements at indices 0..k-1 will be the k smallest.
        
        # Handle case where k might be 0 (if n=1, but we handled that)
        # If n=2, k=1.
        
        # Partition the rows
        # We need to be careful with the inf value. Since inf is the largest, it won't be in the first k
        # unless k >= n-1. But k <= n-1.
        
        # If k == n-1, we are taking all valid distances.
        
        # Let's just use np.sort for simplicity and correctness, n=50 is small.
        # Or np.partition is fine.
        
        # Let's use np.sort to be safe and simple, performance is negligible for n=50.
        sorted_dists = np.sort(dm_work, axis=1)
        
        # Take the first k columns
        # If k is 0, sum is 0.
        if k > 0:
            top_k_dists = sorted_dists[:, :k]
            # Average distance to top k neighbors
            sparsity = np.mean(top_k_dists, axis=1)
        else:
            sparsity = np.zeros(n)
    else:
        sparsity = np.ones(n)
    
    # Ensure sparsity is not zero to avoid division by zero
    sparsity = np.maximum(sparsity, eps)
    
    # Heuristic h[i, j] = prize[j] / sparsity[j]
    # This is a column-wise operation: the value depends only on the destination j.
    # So h[:, j] is the same for all i.
    
    # Create the matrix
    # prize_col shape (n, 1)
    # sparsity_col shape (n, 1)
    # h[i, j] = prize[j] / sparsity[j]
    
    prize_col = prizes.reshape(1, -1)
    sparsity_col = sparsity.reshape(1, -1)
    
    h = prize_col / sparsity_col
    
    # The result is (1, n), broadcast to (n, n)
    h = np.broadcast_to(h, (n, n))
    
    # Ensure positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
