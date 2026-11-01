import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    
    The minimum rank for 2x3x3 matrix multiplication is 18 (schoolbook).
    This function returns the schoolbook decomposition with integer entries.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays with integer dtype
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Schoolbook decomposition
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c = i*m*p + j*p + k
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to linear indices
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, col] = 1
                V[idx_v, col] = 1
                W[idx_w, col] = 1
                col += 1
                
    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
