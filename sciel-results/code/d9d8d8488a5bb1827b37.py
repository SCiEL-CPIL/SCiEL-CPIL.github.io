"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: 9 blocks of 2x2, each multiplied via Strassen's 7-product algorithm.
    
    This uses the 2x2 block decomposition where each block is 2x2, and
    Strassen's algorithm is applied to each 2x2x2x2 sub-problem.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s wall-clock kill

    n, m, p = DIMENSIONS  # all 6

    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                # A block: rows (2*i_block, 2*i_block+1), cols (2*k_block, 2*k_block+1)
                # B block: rows (2*k_block, 2*k_block+1), cols (2*j_block, 2*j_block+1)
                # C block: rows (2*i_block, 2*i_block+1), cols (2*j_block, 2*j_block+1)

                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)

                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)

                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)

                # M1 = (A11 + A22) * (B11 + B22)
                U[u_row(0, 0)][col] = 1
                U[u_row(1, 1)][col] = 1
                V[v_row(0, 0)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M2 = (A21 + A22) * B11
                U[u_row(1, 0)][col] = 1
                U[u_row(1, 1)][col] = 1
                V[v_row(0, 0)][col] = 1
                W[w_row(1, 0)][col] = 1
                W[w_row(1, 1)][col] = -1
                col += 1

                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0)][col] = 1
                V[v_row(0, 1)][col] = 1
                V[v_row(1, 1)][col] = -1
                W[w_row(0, 1)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1)][col] = 1
                V[v_row(1, 0)][col] = 1
                V[v_row(0, 0)][col] = -1
                W[w_row(0, 0)][col] = 1
                W[w_row(1, 0)][col] = 1
                col += 1

                # M5 = (A11 + A12) * B22
                U[u_row(0, 0)][col] = 1
                U[u_row(0, 1)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = -1
                W[w_row(0, 1)][col] = 1
                col += 1

                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0)][col] = 1
                U[u_row(0, 0)][col] = -1
                V[v_row(0, 0)][col] = 1
                V[v_row(0, 1)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1)][col] = 1
                U[u_row(1, 1)][col] = -1
                V[v_row(1, 0)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = 1
                col += 1

    # Verify with random sampling
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    random.seed(42)
    valid = True
    for _ in range(500):
        if time.time() > deadline:
            break
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook (rank 216)
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
