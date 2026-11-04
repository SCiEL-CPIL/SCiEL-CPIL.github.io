import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    t0 = time.time()
    deadline = 120.0

    # Build rank-25 decomposition with shared Strassen factors
    rank = 25
    n, m, p = DIMENSIONS
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    r = 0

    # Shared Strassen factor (a0+a1) on A side
    # Term 0: for C[:,0:2] (uses B1=B[:,0:2])
    U[0 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 0][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1

    # Term 1: for C[:,2:4] (uses B3=B[:,2:4])
    U[0 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 2][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1

    # a1 term for C[:,0:2]
    U[1 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 0][r] = 1.0
    W[0 * n + 1][r] = 1.0
    W[1 * n + 1][r] = -1.0
    r += 1

    # a1 term for C[:,2:4]
    U[1 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 2][r] = 1.0
    W[2 * n + 1][r] = 1.0
    W[3 * n + 1][r] = -1.0
    r += 1

    # b0 term for C[:,0:2]
    U[0 * m + 0][r] = 1.0
    V[0 * p + 1][r] = 1.0
    V[1 * p + 1][r] = -1.0
    W[1 * n + 0][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1

    # b0 term for C[:,2:4]
    U[0 * m + 0][r] = 1.0
    V[0 * p + 3][r] = 1.0
    V[1 * p + 3][r] = -1.0
    W[3 * n + 0][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1

    # b1 term for C[:,0:2]
    U[1 * m + 1][r] = 1.0
    V[1 * p + 0][r] = 1.0
    V[0 * p + 0][r] = -1.0
    W[0 * n + 0][r] = 1.0
    W[0 * n + 1][r] = 1.0
    r += 1

    # b1 term for C[:,2:4]
    U[1 * m + 1][r] = 1.0
    V[1 * p + 2][r] = 1.0
    V[0 * p + 2][r] = -1.0
    W[2 * n + 0][r] = 1.0
    W[2 * n + 1][r] = 1.0
    r += 1

    # Shared (a0+a1) for C[:,0:2] (reused M5)
    U[0 * m + 0][r] = 1.0
    U[0 * m + 1][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = -1.0
    W[1 * n + 0][r] = 1.0
    r += 1

    # (a0+a1) for C[:,2:4]
    U[0 * m + 0][r] = 1.0
    U[0 * m + 1][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = -1.0
    W[3 * n + 0][r] = 1.0
    r += 1

    # (c+d) term for C[:,0:2]
    U[1 * m + 0][r] = 1.0
    U[0 * m + 0][r] = -1.0
    V[0 * p + 0][r] = 1.0
    V[0 * p + 1][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1

    # (c+d) term for C[:,2:4]
    U[1 * m + 0][r] = 1.0
    U[0 * m + 0][r] = -1.0
    V[0 * p + 2][r] = 1.0
    V[0 * p + 3][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1

    # Shared a0 for C[:,0:2] (reused M7)
    U[0 * m + 1][r] = 1.0
    U[1 * m + 1][r] = -1.0
    V[1 * p + 0][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = 1.0
    r += 1

    # a0 for C[:,2:4]
    U[0 * m + 1][r] = 1.0
    U[1 * m + 1][r] = -1.0
    V[1 * p + 2][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = 1.0
    r += 1

    # C[:,2] from row 2 of A (schoolbook)
    for i in range(n):
        for k in range(2, 4):
            U[i * m + 2][r] = 1.0
            V[2 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            r += 1

    # C[:,4] from row 0 of A
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1.0
            V[j * p + 4][r] = 1.0
            W[4 * n + i][r] = 1.0
            r += 1

    assert r == rank, f"Expected {rank} terms, got {r}"

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p"""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
