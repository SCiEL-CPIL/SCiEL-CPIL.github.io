import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m (since we can decompose each of the m terms A[i,j]B[j,k] with rank 1)
    rank = m
    
    # Initialize U, V, W as lists of lists of zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U index: i*m + j, col: j
        for i in range(n):
            U[i * m + j][j] = 1.0
            
        # V index: j*p + k, col: j
        for k in range(p):
            V[j * p + k][j] = 1.0
            
        # W index: k*n + i, col: j
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
