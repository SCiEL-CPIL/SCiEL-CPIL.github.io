import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Schoolbook decomposition
    # T[i, j, k] = sum_l A[i, l] * B[l, k]
    # Term for (i, j, k):
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists to match the required interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
