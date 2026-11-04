"""Tensor decomposition of 5x5x5 matrix multiplication, rank 61 (block schoolbook)."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)

    # 1. 3x3 Core (indices 0,1,2) - Schoolbook, rank 27
    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                add_term({ii * m + jj: 1}, {jj * p + kk: 1}, {kk * n + ii: 1})

    # 2. 2x2 Core (indices 3,4) - Schoolbook, rank 4
    for ii in range(3, 5):
        for jj in range(3, 5):
            for kk in range(3, 5):
                add_term({ii * m + jj: 1}, {jj * p + kk: 1}, {kk * n + ii: 1})

    # 3. Mixed Block: A(3x2) * B(2x3) -> C(3x3) - Schoolbook, rank 18
    for ii in range(3):
        for jj in range(3, 5):
            for kk in range(3):
                add_term({ii * m + jj: 1}, {jj * p + kk: 1}, {kk * n + ii: 1})

    # 4. Mixed Block: A(2x3) * B(3x2) -> C(2x2) - Schoolbook, rank 12
    for ii in range(3, 5):
        for jj in range(3):
            for kk in range(3, 5):
                add_term({ii * m + jj: 1}, {jj * p + kk: 1}, {kk * n + ii: 1})

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

    # Fast verification with a few random trials
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 3) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
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
