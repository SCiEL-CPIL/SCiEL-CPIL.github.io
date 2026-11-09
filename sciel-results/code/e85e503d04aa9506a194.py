"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (9, rank), (9, rank), (9, rank).
    The known optimal rank for 3x3 matrix multiplication is 23.
    However, constructing the explicit matrices for rank 23 is complex.
    We return the schoolbook decomposition with rank 27, which is guaranteed correct.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 9, 9
    
    # Schoolbook decomposition (rank 27)
    rank = n * m * p  # 27
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
