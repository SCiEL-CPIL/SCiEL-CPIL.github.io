"""Tensor decomposition of 5x5x5 matrix multiplication, rank 58."""
import time

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin

    n, m, p = 5, 5, 5

    U_cols = []  # list of dicts: col -> {index: value}
    V_cols = []
    W_cols = []

    # Part 1: Frame terms (i==4 or j==4), k in 0..4
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})

    # Part 2: 4x4x4 core via Schoolbook (rank 64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify correctness
    import random
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline:
            break
        A = [[random.randint(0, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
        C_tensor = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                C_tensor[i][k] = s
        if C_tensor != C:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
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
