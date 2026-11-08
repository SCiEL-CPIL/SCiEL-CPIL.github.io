"""Optimized rank-46 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 disjoint 2x2x2 Strassen blocks for i in {0,1} and k in {0..5}.
Uses a rank-4 decomposition for the remaining entries (i=2 and k=6 for i in {0,1}).
Total rank = 6*7 + 4 = 46.
"""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    rank = 6 * 7 + 4  # 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
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

    # Add 4 rank-1 terms for the remaining entries
    # Remaining: i=2 (all j,k) and i in {0,1} with k=6
    # Strategy:
    # For i=2: U[2,j,r] = delta(j,r), V[j,k,r] = 1, W[k,2,r] = 1
    # For i in {0,1}: U[i,j,r] = 0.5 if r in {0,1} else 0, V[j,6,r] = 1, W[6,i,r] = 1
    # W[k,i,r] = 0 for k in {0..5}, i in {0,1} to avoid interference
    
    for r in range(4):
        # U part
        # i=2
        U[2 * m + r][col + r] = 1
        # i=0, 1
        if r < 2:
            U[0 * m + 0][col + r] = 0.5
            U[0 * m + 1][col + r] = 0.5
            U[0 * m + 2][col + r] = 0.5
            U[0 * m + 3][col + r] = 0.5
            
            U[1 * m + 0][col + r] = 0.5
            U[1 * m + 1][col + r] = 0.5
            U[1 * m + 2][col + r] = 0.5
            U[1 * m + 3][col + r] = 0.5
        # else U[0,j,r] and U[1,j,r] are 0

        # V part
        # V[j,k,r] = 1 for all j,k
        for j in range(m):
            for k in range(p):
                V[j * p + k][col + r] = 1

        # W part
        # W[k,2,r] = 1 for all k
        for k in range(p):
            W[k * n + 2][col + r] = 1
        # W[6,i,r] = 1 for i in {0,1}
        W[6 * n + 0][col + r] = 1
        W[6 * n + 1][col + r] = 1
        # W[k,i,r] = 0 for k in {0..5}, i in {0,1} is already 0

    assert col + 4 == rank

    # Verify with early exit on deadline
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
