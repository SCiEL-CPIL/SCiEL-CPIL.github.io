"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def construct():
    """
    Return the exact rank-18 (schoolbook) tensor decomposition for 2x3x3 matrix multiplication.
    The tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k.
    Schoolbook: one rank-1 term per (i,j,k) triple.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2*3*3 = 18

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
