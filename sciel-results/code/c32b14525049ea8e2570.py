"""
Decomposition of the 3x3x5 matrix multiplication tensor.
Uses the schoolbook (naive) algorithm which has rank n*m*p.
"""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor.
    The decomposition is rank 45 (schoolbook).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over all indices i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1.0
                # V index: j*p + k
                V[j * p + k][col] = 1.0
                # W index: k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
