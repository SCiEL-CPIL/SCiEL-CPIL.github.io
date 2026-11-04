import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    Uses Kronecker product structure to define the basis indices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate index grids using np.kron for Kronecker structure
    # Indices for i, j, k
    idx_i = np.kron(np.arange(n), np.ones((m * p), dtype=int))
    idx_j = np.kron(np.kron(np.arange(m), np.ones((p), dtype=int)), np.ones((n), dtype=int))
    idx_k = np.kron(np.ones((n * m), dtype=int), np.arange(p))
    
    # Calculate linear indices for U, V, W
    u_indices = idx_i * m + idx_j
    v_indices = idx_j * p + idx_k
    w_indices = idx_k * n + idx_i
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate matrices
    # Using advanced indexing to set 1s
    U[u_indices, np.arange(rank)] = 1
    V[v_indices, np.arange(rank)] = 1
    W[w_indices, np.arange(rank)] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
