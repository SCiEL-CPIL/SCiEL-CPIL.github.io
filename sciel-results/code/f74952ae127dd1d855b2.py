import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of the trivial decomposition
    # We create index arrays for all combinations of (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    
    # Generate all indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D arrays
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate column indices in U, V, W
    # U index: i * m + j
    u_cols = i_flat * m + j_flat
    # V index: j * p + k
    v_cols = j_flat * p + k_flat
    # W index: k * n + i
    w_cols = k_flat * n + i_flat
    
    # The column index in the rank dimension is just 0..rank-1
    rank_idx = np.arange(rank)
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Set values to 1.0
    U[u_cols, rank_idx] = 1.0
    V[v_cols, rank_idx] = 1.0
    W[w_cols, rank_idx] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
