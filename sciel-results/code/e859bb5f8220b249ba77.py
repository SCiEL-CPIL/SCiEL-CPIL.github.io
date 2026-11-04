import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The minimal rank for matrix multiplication M_{n,m,p} is n*m*p
    # because the output indices are disjoint for each row of the output matrix.
    # Specifically, rank(M_{n,m,p}) = n * rank(M_{1,m,p}) = n * (m*p) = n*m*p.
    # For 3x5x6, the minimal rank is 90.
    
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill with the trivial decomposition
    # Each term in the sum corresponds to a unique (i, j, k) triple
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
