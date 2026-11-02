"""Matrix multiplication 2,2,16 - rank-minimal exact tensor decomposition."""

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 2,2,16 - rank-minimal exact tensor decomposition.
    Rank is 64 (16 copies of rank-4 M_{2,2,1}).
    """
    import numpy as np

    n, m, p = 2, 2, 16
    rank = n * m * p  # 64

    # Generate indices for all (i, j, k) combinations
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Total rank = n * m * p
    
    # We can create the indices using meshgrid or simple logic
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Create arrays of indices
    i_idx, j_idx, k_idx = np.indices((n, m, p)).reshape(3, -1)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s
    # Columns are 0..rank-1
    cols = np.arange(rank)
    
    U[u_idx, cols] = 1
    V[v_idx, cols] = 1
    W[w_idx, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
