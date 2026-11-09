import time
import random
import itertools

DIMENSIONS = [3, 3, 5]
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15

    def schoolbook():
        rank = nm * mp
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # The schoolbook decomposition always works with integer entries.
    result = schoolbook()
    return result
# EVOLVE-BLOCK-END
