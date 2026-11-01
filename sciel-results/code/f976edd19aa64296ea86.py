"""Matrix multiplication 2,2,13 - structured schoolbook decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    total = n * m * p
    U = np.zeros((n * m, total))
    V = np.zeros((m * p, total))
    W = np.zeros((p * n, total))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
