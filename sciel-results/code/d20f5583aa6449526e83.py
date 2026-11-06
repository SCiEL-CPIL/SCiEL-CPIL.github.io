"""
Strategy: Fiedler Vector Initialization + Vectorized Local Search.
Instead of using the top eigenvector of the adjacency matrix (which can be degenerate or less effective for cut quality), 
this approach uses the Fiedler vector (eigenvector corresponding to the second-smallest eigenvalue of the graph Laplacian) 
to generate an initial 0/1 labeling. This leverages global graph structure for a high-quality start near a good basin.
The initial partition is then refined with a vectorized 1-flip local search that repeatedly flips the vertex with the 
largest positive gain until no improvement is possible or the time limit is reached.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Fiedler vector (Laplacian spectral) initialization followed by 1-flip local search.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    # Check if A is all zeros
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    # 1. Fiedler Vector Initialization
    # Build the graph Laplacian: L = D - A
    # D is the degree matrix
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Compute eigenvectors of L using numpy
    # We want the eigenvector corresponding to the second-smallest eigenvalue (Fiedler vector)
    # For efficiency and stability, we use np.linalg.eigh which works for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort eigenvalues and their corresponding eigenvectors
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # The smallest eigenvalue should be 0 (for a connected graph).
        # The second smallest is the Fiedler value.
        # The Fiedler vector is eigenvectors[:, 1]
        fiedler_vec = eigenvectors[:, 1]
        
        # Round to 0/1 based on sign
        # fiedler_vec > 0 -> 1, else 0
        labels = (fiedler_vec > 0).astype(np.int8)
        
    except Exception as e:
        # Fallback to random initialization if eigendecomposition fails
        np.random.seed(42)
        labels = np.random.randint(0, 2, n).astype(np.int8)
    
    # 2. Vectorized 1-Flip Local Search
    # We repeatedly find the vertex with the maximum gain from flipping and flip it
    # until no positive gain exists or time runs out.
    
    # Precompute degrees
    degrees = np.sum(adj, axis=1)
    
    # We'll maintain the current labels and compute gains efficiently
    # Gain for flipping vertex i = 2 * (number of neighbors of i on the same side) - degree[i]
    # If gain > 0, flipping improves the cut.
    
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        
        # Compute for each vertex i: count of neighbors j where labels[j] == labels[i]
        # same_side[i] = sum_j adj[i,j] * (labels[i] == labels[j])
        # We can vectorize this:
        # same_mask[i, j] = (labels[i] == labels[j])
        # same_counts[i] = sum_j adj[i, j] * same_mask[i, j]
        
        same_mask = (labels[:, None] == labels[None, :])
        same_counts = np.sum(adj * same_mask, axis=1)
        
        gains = 2 * same_counts - degrees
        
        max_gain = np.max(gains)
        if max_gain > 1e-9:
            flip_idx = np.argmax(gains)
            labels[flip_idx] = 1 - labels[flip_idx]
            improved = True
    
    return labels
# EVOLVE-BLOCK-END
