import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p
    
    # Create index arrays for i, j, k
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Create U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Calculate row indices for U, V, W
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Set the 1s
    # Each column r corresponds to the r-th (i,j,k) triplet
    # U[u_rows[r], r] = 1
    # V[v_rows[r], r] = 1
    # W[w_rows[r], r] = 1
    
    # We can use advanced indexing
    cols = np.arange(rank)
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
