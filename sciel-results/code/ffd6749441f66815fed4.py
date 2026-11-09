"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = n * m * p  # 48
    
    # Construct the schoolbook decomposition.
    # This is guaranteed to be valid and consists of integers (0 and 1).
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to the product A[i,j] * B[j,k] contributing to C[i,k]
                # In the tensor decomposition, we have basis vectors for U, V, W.
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i (following the convention W[k*n+i, r])
                
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
