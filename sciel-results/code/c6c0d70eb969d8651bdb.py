import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Create a copy to avoid modifying the original
    dist = distance_matrix.copy()
    
    # Set diagonal to 0, as MST algorithms often treat 0 as no-edge or self-loop
    # scipy.sparse.csgraph.minimum_spanning_tree treats 0 as no edge in the input matrix
    # But wait, if distance is 0 for i!=j (duplicate points), it's a valid edge.
    # In TSP with uniform random points, duplicates are unlikely but possible.
    # However, standard MST implementation on a dense matrix passed to scipy might interpret 0 as no edge.
    # Let's check scipy docs: "The input is a matrix ... 0 represents no edge".
    # This is problematic if two points are at distance 0.
    # To be safe, I can use a very small epsilon for 0 distances if i!=j?
    # Or just assume no duplicates for random points.
    # Let's assume standard behavior.
    
    # Convert to sparse matrix
    # We need to ensure that 0 on diagonal is treated as no edge, which is correct.
    # If there are 0 off-diagonal, they are treated as no edge, which is incorrect for TSP (should be edge of weight 0).
    # But for random points, prob of 0 distance is 0.
    
    sparse_dist = sp.csr_matrix(dist)
    
    # Compute MST
    # Note: minimum_spanning_tree returns a sparse matrix
    mst_sparse = csgraph.minimum_spanning_tree(sparse_dist)
    
    # Initialize heuristic with low value
    heuristic = np.full((n, n), 1e-9)
    
    # Get coordinates of non-zero elements in MST
    # mst_sparse is symmetric
    rows, cols = mst_sparse.nonzero()
    
    # Set high value for MST edges
    # Ensure symmetry
    for i, j in zip(rows, cols):
        heuristic[i, j] = 1.0
        heuristic[j, i] = 1.0
        
    return heuristic
