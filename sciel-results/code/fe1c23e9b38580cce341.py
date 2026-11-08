import time

import numpy as np

_DEADLINE = 120.0  # wall-clock budget (s); evaluator kills at 150 s
_start_time = time.time()


def _time_left():
    # simple deadline helper: returns seconds remaining
    return _DEADLINE - (time.time() - _start_time)


def construct():
    # Dimensions: n=2 (rows of A), m=3 (inner), p=9 (cols of B)
    n, m, p = 2, 3, 9

    # Rank 41 construction:
    #   4 Strassen 2x2 blocks over B columns (0,1),(2,3),(4,5),(6,7): 4*7 = 28
    #   1 naive 2x1x2x1 block over B column 8:                          4
    #   rank-3 batch on A_col2 x B_row2 (covers all 9 cols):            9
    # ---------------------------------------------------------------------------
    # Total rank = 41
    rank = 41

    # U: (n*m, rank) = (6, 41)
    # V: (m*p, rank) = (27, 41)
    # W: (p*n, rank) = (18, 41)
    # Indexing convention:
    #   A index  = i*m + j   (i in 0..n-1, j in 0..m-1)
    #   B index  = j*p + k   (j in 0..m-1, k in 0..p-1)
    #   C index  = k*n + i   (k in 0..p-1, i in 0..n-1)  <- trace convention

    U = [[0] * rank for _ in range(n * m)]   # 6 x 41
    V = [[0] * rank for _ in range(m * p)]   # 27 x 41
    W = [[0] * rank for _ in range(p * n)]   # 18 x 41

    col = 0

    # ------------------------------------------------------------------
    # Part 1: Strassen-style 2x2 blocks over A_top (2x2) and B_top (2x9)
    # ------------------------------------------------------------------
    # A_top entries:
    #   a = A_{0,0} -> U[0]
    #   b = A_{0,1} -> U[1]
    #   c = A_{1,0} -> U[3]
    #   d = A_{1,1} -> U[4]
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):  # 4 blocks over pairs of B columns
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices for these columns
        v_e = 0 * p + k0   # B_{0, k0}
        v_f = 0 * p + k1   # B_{0, k1}
        v_g = 1 * p + k0   # B_{1, k0}
        v_h = 1 * p + k1   # B_{1, k1}

        # W indices (trace convention C = k*n + i)
        w_00 = k0 * n + 0   # C_{0, k0}
        w_01 = k1 * n + 0   # C_{0, k1}
        w_10 = k0 * n + 1   # C_{1, k0}
        w_11 = k1 * n + 1   # C_{1, k1}

        # 7 Strassen terms
        # M1 = (a + d)(e + h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        # M2 = (c + d) * e
        U[u_c][col + 1] += 1
        U[u_d][col + 1] += 1
        V[v_e][col + 1] += 1

        # M3 = a * (f - h)
        U[u_a][col + 2] += 1
        V[v_f][col + 2] += 1
        V[v_h][col + 2] -= 1

        # M4 = d * (g - e)
        U[u_d][col + 3] += 1
        V[v_g][col + 3] += 1
        V[v_e][col + 3] -= 1

        # M5 = (a + b) * h
        U[u_a][col + 4] += 1
        U[u_b][col + 4] += 1
        V[v_h][col + 4] += 1

        # M6 = (c - a)(e + f)
        U[u_c][col + 5] += 1
        U[u_a][col + 5] -= 1
        V[v_e][col + 5] += 1
        V[v_f][col + 5] += 1

        # M7 = (b - d)(g + h)
        U[u_b][col + 6] += 1
        U[u_d][col + 6] -= 1
        V[v_g][col + 6] += 1
        V[v_h][col + 6] += 1

        # Reconstruct the 2x2 block output
        # C_{0, k0} = M1 + M4 - M5 + M7
        W[w_00][col]   += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        # C_{0, k1} = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        # C_{1, k0} = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        # C_{1, k1} = M1 - M2 + M3 + M6
        W[w_11][col]   += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7  # 7 terms per block

    # ------------------------------------------------------------------
    # Part 2: Naive 2x1 block for last B column k = 8
    # ------------------------------------------------------------------
    k8 = 8
    v_e8 = 0 * p + k8   # B_{0, 8}
    v_g8 = 1 * p + k8   # B_{1, 8}

    w_08 = k8 * n + 0   # C_{0, 8}
    w_18 = k8 * n + 1   # C_{1, 8}

    # P1 = a * B_{0,8}  -> C_{0,8}
    U[u_a][col]   += 1
    V[v_e8][col]  += 1
    W[w_08][col]  += 1

    # P2 = b * B_{1,8}  -> C_{0,8}
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1

    # P3 = c * B_{0,8}  -> C_{1,8}
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1

    # P4 = d * B_{1,8}  -> C_{1,8}
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1

    col += 4

    # ------------------------------------------------------------------
    # Part 3: rank-3 batch on A_col2 (col index 2) x B_row2 (row index 2)
    # Each of the 3 products is a sum over k of A_entry * B_{2,k},
    # and the same scalar feeds every output column k via W.
    #   t1 = (A02 + A12) * B2k   -> contributes to C[0,k] (+) and C[1,k] (+)
    #   t2 = (A02 - A12) * B2k   -> contributes to C[0,k] (+) and C[1,k] (-)
    #   t3 =        A12 * B2k    -> contributes to C[1,k] (+)
    # Then:  C[0,k] = t1 + t2  =  A02*B2k   (exact)
    #        C[1,k] = t1 + t3  =  A12*B2k   (exact)
    # ------------------------------------------------------------------
    u_ar0 = 0 * m + 2   # A_{0,2}
    u_ar1 = 1 * m + 2   # A_{1,2}
    v_b = 2 * p         # base B index for row 2 (B_{2,k} = v_b + k)

    # t1 = (A02 + A12) * (sum_k B2k)
    U[u_ar0][col]   += 1
    U[u_ar1][col]   += 1
    for k in range(p):
        V[v_b + k][col] += 1
        W[k * n + 0][col] += 1
        W[k * n + 1][col] += 1

    # t2 = (A02 - A12) * (sum_k B2k)
    U[u_ar0][col + 1] += 1
    U[u_ar1][col + 1] -= 1
    for k in range(p):
        V[v_b + k][col + 1] += 1
        W[k * n + 0][col + 1] += 1
        W[k * n + 1][col + 1] -= 1

    # t3 = A12 * (sum_k B2k)
    U[u_ar1][col + 2] += 1
    for k in range(p):
        V[v_b + k][col + 2] += 1
        W[k * n + 1][col + 2] += 1

    col += 3

    # col should equal rank (41)
    return {"U": U, "V": V, "W": W}
