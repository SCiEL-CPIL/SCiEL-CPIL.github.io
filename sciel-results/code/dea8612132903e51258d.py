import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We construct the factors using the standard schoolbook algorithm
    # but implemented with vectorized operations for a different code structure.
    
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Each column c in U, V, W corresponds to a unique (i, j, k) triple.
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # We can generate the indices for i, j, k
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten the indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Column indices are just 0 to rank-1
    col_idx = np.arange(rank)
    
    # Row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Set the ones
    U[u_rows, col_idx] = 1
    V[v_rows, col_idx] = 1
    W[w_rows, col_idx] = 1
    
    # Convert to lists as required by the interface (implied by seed, though numpy arrays are often accepted, 
    # the seed returns lists of lists. To be safe, I will return lists of lists of ints.)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
