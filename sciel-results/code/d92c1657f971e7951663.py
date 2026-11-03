import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = n * m * p
    
    # Create index grids
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D arrays
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate linear indices for U, V, W rows
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Column index is just 0 to rank-1
    cols = np.arange(rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
