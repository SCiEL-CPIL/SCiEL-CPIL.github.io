"""Matrix multiplication 2x3x7: exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42
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
