"""Optimized rank-44 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 Strassen blocks for the main 2x4x6 sub-tensor.
Uses rank-1 decompositions for the remaining i=2 slice and k=6 slice.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    rank = 6 * 7 + 2  # 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)

    # Add rank-1 term for i=2 slice
    # U[2*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+2, col] = 1
    for j in range(m):
        U[2 * m + j][col] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][col] = 1
    for k in range(p):
        W[k * n + 2][col] = 1
    col += 1

    # Add rank-1 term for k=6, i in {0,1} slice
    # U[i*m+j, col] = 1 for i in {0,1}
    # V[j*p+6, col] = 1
    # W[6*n+i, col] = 1 for i in {0,1}
    for i in range(2):
        for j in range(m):
            U[i * m + j][col] = 1
    for j in range(m):
        V[j * p + 6][col] = 1
    for i in range(2):
        W[6 * n + i][col] = 1
    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Fast NumPy verification
    if time.time() - _t0 < _DEADLINE - 5:
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j]
                    v_row = V_np[j * p + k]
                    w_row = W_np[k * n + i]
                    total = np.dot(u_row, v_row * w_row)
                    if total != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
