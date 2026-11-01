import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices for U, V, W rows
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    u_rows = i_idx.flatten() * m + j_idx.flatten()
    v_rows = j_idx.flatten() * p + k_idx.flatten()
    w_rows = k_idx.flatten() * n + i_idx.flatten()
    
    # Initialize zero matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Set the 1s using advanced indexing
    cols = np.arange(rank)
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
