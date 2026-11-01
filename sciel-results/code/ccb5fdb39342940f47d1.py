"""Matrix multiplication tensor decomposition for 2x5 by 5x5."""
DIMENSIONS = [2, 5, 5]

import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Using a vectorized construction approach
    # This generates the same trivial decomposition but via tensor operations
    rank = n * m * p
    
    # Create indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices to map to rank
    r_flat = (i_idx * m * p + j_idx * p + k_idx).flatten()
    
    # U shape: (n*m, rank)
    # Row index: i*m + j
    u_rows = (i_idx * m + j_idx).flatten()
    
    # V shape: (m*p, rank)
    # Row index: j*p + k
    v_rows = (j_idx * p + k_idx).flatten()
    
    # W shape: (p*n, rank)
    # Row index: k*n + i
    w_rows = (k_idx * n + i_idx).flatten()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[u_rows, r_flat] = 1
    V[v_rows, r_flat] = 1
    W[w_rows, r_flat] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
