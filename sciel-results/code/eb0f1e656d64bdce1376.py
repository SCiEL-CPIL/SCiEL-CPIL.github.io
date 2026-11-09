import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for vectorized assignment
    # c ranges from 0 to rank-1
    c = np.arange(rank)
    
    # Decompose c into i, j, k
    # c = i * m * p + j * p + k
    i = (c // (m * p)) % n
    j = (c // p) % m
    k = c % p
    
    # Flatten indices for U, V, W
    u_idx = i * m + j
    v_idx = j * p + k
    w_idx = k * n + i
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set values using advanced indexing
    U[u_idx, c] = 1
    V[v_idx, c] = 1
    W[w_idx, c] = 1
    
    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
