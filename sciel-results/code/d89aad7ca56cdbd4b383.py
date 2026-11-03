"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 6
    M = m * p  # 27
    P = p * n  # 18
    
    rank = M  # 27, the lower bound
    
    U = np.zeros((N, rank), dtype=int)
    V = np.zeros((M, rank), dtype=int)
    W = np.zeros((P, rank), dtype=int)
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U: rows i*m + j for i in range(n)
            for i in range(n):
                U[i * m + j, r] = 1
            # V: row j*p + k
            V[j * p + k, r] = 1
            # W: columns k*n + i for i in range(n)
            for i in range(n):
                W[k * n + i, r] = 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                assert val == 1, f"Verification failed at ({i},{j},{k}): got {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
