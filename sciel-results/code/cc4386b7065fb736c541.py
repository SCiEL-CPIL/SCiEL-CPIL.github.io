"""Program for 3x4x7 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy: Use the schoolbook decomposition as baseline (rank 84),
    then attempt to improve by exploiting structure.
    
    For now, return the schoolbook decomposition which is guaranteed correct.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 84
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 12 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 21 x rank
    
    # Schoolbook: for each (i, j, k), one rank-1 term
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
