"""Matrix multiplication 3x3x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
n, m, p = DIMENSIONS

# Deadline guard
START_TIME = time.time()
DEADLINE = 150.0  # seconds, evaluator kills at 150s

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    # Schoolbook decomposition as fallback
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
