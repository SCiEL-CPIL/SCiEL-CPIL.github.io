"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    Uses schoolbook with rank 48.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

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
