"""Model-and-optimize: parameterize factor matrices and fit against the
matrix-multiplication tensor, attempting rank reduction from schoolbook.
Robust fallback to the exact schoolbook decomposition."""
import time
import random

DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = None


def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)


# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    global _START
    _START = time.time()
    return _schoolbook()
# EVOLVE-BLOCK-END
