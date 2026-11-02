import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook algorithm: Rank = n * m * p
    # This is the standard decomposition where each term corresponds to a specific (i, j, k)
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j, col] = 1
                U[i * m + j, col] = 1
                # V[j, k, col] = 1
                V[j * p + k, col] = 1
                # W[k, i, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
