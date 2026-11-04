import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # The matrix multiplication tensor for n x m * m x p
    # decomposes into a direct sum of m tensors, one for each intermediate index j.
    # Each component tensor T_j corresponds to the outer product of a (n x 1) slice of A
    # and a (1 x p) slice of B, resulting in a rank n*p tensor.
    # Since the supports of these component tensors in the V-mode (index j) are disjoint,
    # the total rank is the sum of the ranks of the components.
    # Total Rank = m * (n * p) = n * m * p.
    
    total_rank = n * m * p
    
    U = np.zeros((dim_U, total_rank))
    V = np.zeros((dim_V, total_rank))
    W = np.zeros((dim_W, total_rank))
    
    # Vectorized construction of the trivial decomposition
    # We iterate over the 'column' of the factor matrices, which corresponds to a triple (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    
    # Create arrays of indices for vectorized assignment
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate flat indices for U, V, W
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # The rank is total_rank, so we assign 1.0 to the corresponding positions
    # Each (i,j,k) triple corresponds to a unique column in the factor matrices
    # Since the mapping (i,j,k) -> column is bijective (size n*m*p), we can just set them to 1.
    
    # Initialize columns with 0, then set specific entries to 1
    # Actually, since it's a permutation of indices, we can just create a sparse-like structure
    # But numpy dense is fine for rank 75.
    
    # We can simply set U[u_idx, col] = 1, etc.
    # However, we need to map each triple to a unique column index.
    # The triples are generated in lexicographic order, which is a bijection to 0..total_rank-1.
    
    cols = np.arange(total_rank)
    
    U[u_idx, cols] = 1.0
    V[v_idx, cols] = 1.0
    W[w_idx, cols] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
