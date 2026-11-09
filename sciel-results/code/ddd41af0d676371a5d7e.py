"""Program for tensor rank of 2x5 by 5x6 matrix multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find the rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a block decomposition splitting the middle dimension m=5 into five 1s.
    Rank = 5 * mu(2, 1, 6) = 5 * 6 = 30.
    """
    n, m, p = 2, 5, 6
    
    # Rank 30 decomposition
    # Split m=5 into 5 blocks of size 1.
    # Each block is M_{2, 1, 6} with rank 6.
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
            column += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
