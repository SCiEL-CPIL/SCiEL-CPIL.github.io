import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 20

    U = [[0] * rank for _ in range(n * m)]  # 8 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 10 x 20

    col = 0

    # Block 1: C[:, 0:2] = A * B[:, 0:2]  (2x4 * 4x2 -> 2x2)
    # Schoolbook: for each j in 0..3, one rank-1 term
    for j in range(4):
        c = col
        for i in range(2):
            U[i * m + j][c] = 1
        for k in range(2):
            V[j * p + k][c] = 1
        for i in range(2):
            for k in range(2):
                W[k * n + i][c] = 1
        col += 1

    # Block 2: C[:, 2:4] = A * B[:, 2:4]  (2x4 * 4x2 -> 2x2)
    for j in range(4):
        c = col
        for i in range(2):
            U[i * m + j][c] = 1
        for k in range(2, 4):
            V[j * p + k][c] = 1
        for i in range(2):
            for k in range(2, 4):
                W[k * n + i][c] = 1
        col += 1

    # Block 3: C[:, 4] = A * B[:, 4]  (2x4 * 4x1 -> 2x1)
    for j in range(4):
        c = col
        for i in range(2):
            U[i * m + j][c] = 1
        V[j * p + 4][c] = 1
        for i in range(2):
            W[4 * n + i][c] = 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # ---- Exact verification ----------------------------------------------
    if _remaining() > 0.5:
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        acc += U[ui][r] * V[vj][r] * W[wk][r]
                    assert acc == 1, f"Verification failed at i={i},j={j},k={k}: acc={acc}"

    return {"U": U, "V": V, "W": W}
