"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    def elapsed():
        return time.time() - start_time

    def schoolbook():
        """Rank n*m*p exact decomposition (always valid)."""
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

    # Block-Strassen family: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)

    return schoolbook()
