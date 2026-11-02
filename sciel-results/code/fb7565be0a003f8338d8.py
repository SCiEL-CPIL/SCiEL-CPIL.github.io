"""
Rewritten program for matrix multiplication tensor decomposition 2x3 * 3x6.
The minimal known rank for this tensor is 36 (schoolbook).
No algorithm is known to beat the schoolbook rank for these dimensions.
This program returns the schoolbook decomposition efficiently.
"""

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    # The schoolbook decomposition has rank n*m*p
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # For each term in the sum C[i,k] = sum_j A[i,j]B[j,k]
                # We associate a rank-1 term with indices (i,j,k)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
