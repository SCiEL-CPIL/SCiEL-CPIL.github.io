"""Matrix multiplication 4x4x5 - exact schoolbook with Fraction verification."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = n * m * p  # 80

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

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
