import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U_22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_22 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_22 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Kronecker product to get 4x4x4 decomposition (rank 49)
    U_44 = np.kron(U_22, U_22)
    V_44 = np.kron(V_22, V_22)
    W_44 = np.kron(W_22, W_22)
    
    # Assemble for 4x4x8 (rank 98)
    # U is 16x98
    # V is 32x98
    # W is 32x98
    
    rank = 98
    
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half (cols 0-3 of C, rows 0-15 of B)
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # Second half (cols 4-7 of C, rows 16-31 of B)
    U[:, 49:98] = U_44
    V[16:32, 49:98] = V_44
    W[16:32, 49:98] = W_44
    
    # Convert to list of lists for output
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
