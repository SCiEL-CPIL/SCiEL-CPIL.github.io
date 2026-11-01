"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W index is k*n + i
    
    # Return the schoolbook decomposition (rank 36).
    # This is guaranteed correct. Finding a lower rank (e.g. 34 or 32) requires
    # specific hardcoded matrices or a successful search, which is not feasible
    # within the current constraints without prior knowledge of the decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
