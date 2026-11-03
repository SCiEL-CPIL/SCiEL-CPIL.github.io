"""Program for rank-minimal exact tensor decomposition of 3x3x5 matrix multiplication."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    """
    Returns a rank-45 (schoolbook) exact decomposition for 3x3 @ 3x5 matrix multiplication.
    The tensor is T[i,j,k; i',j',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}.
    We use the known schoolbook rank n*m*p = 45 as a valid baseline.
    """
    n, m, p = 3, 3, 5
    rank = n * m * p  # 45
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * rank for _ in range(p * n)]  # 15 x 45
    
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
