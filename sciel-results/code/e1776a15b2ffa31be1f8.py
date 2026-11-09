import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 25
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # For each (j,k) pair, one rank term
            # U[i*m + j, col] = 1 for all i
            for i in range(n):
                U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            # W[k*n + i, col] = 1 for all i
            for i in range(n):
                W[k * n + i, col] = 1
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
