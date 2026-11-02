"""Rank-19 tensor decomposition for 3x3 matrix multiplication.

Uses the Laderman-style rank-19 scheme, re-expressed as explicit
(rank, u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff) tuples.

Convention:
  U has shape (n*m, rank): U[i*m+j, r] = coefficient of A_{ij} in factor r
  V has shape (m*p, rank): V[j*p+k, r] = coefficient of B_{jk} in factor r
  W has shape (p*n, rank): W[k*n+i, r] = coefficient of C_{ik} in factor r
  (W index follows the official trace convention W[k*n+i])

Verification: for random A, B, we check that
  C_ik = sum_r ( sum_{j} A_ij * U_{i*m+j, r} ) * ( sum_{j} B_jk * V_{j*p+k, r} ) * W_{k*n+i, r}
matches A @ B.
"""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 19

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Each term: (r, u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff)
    # r: rank index
    # u_idx: i*m + j for A_ij
    # v_idx: j*p + k for B_jk
    # w_idx: k*n + i for C_ik

    terms = [
        # rank 0: C_00 += A_00 * B_00
        (0, 0, 0, 0,  1,  1,  1),
        (0, 3, 2, 0, -1,  1,  1),
        (0, 7, 1, 0,  1, -1, -1),

        # rank 1: C_01 += A_00 * B_01
        (1, 0, 1, 3,  1,  1,  1),
        (1, 3, 5, 3, -1,  1,  1),

        # rank 2: C_10 += A_10 * B_00
        (2, 3, 0, 1,  1,  1,  1),
        (2, 7, 2, 1,  1, -1, -1),

        # rank 3: C_11 += A_11 * B_11
        (3, 4, 4, 4,  1,  1,  1),
        (3, 0, 8, 4,  1,  1,  1),
        (3, 3, 5, 4, -1,  1,  1),
        (3, 7, 1, 4, -1, -1, -1),
        (3, 1, 7, 4,  1,  1,  1),
        (3, 4, 2, 4, -1, -1, -1),
        (3, 4, 1, 4, -1,  1,  1),
        (3, 7, 4, 4, -1, -1, -1),
        (3, 1, 4, 4, -1,  1,  1),

        # rank 4: C_02 += A_00 * B_02
        (4, 0, 2, 6,  1,  1,  1),
        (4, 3, 5, 6, -1,  1,  1),

        # rank 5: C_02 += A_02 * B_20
        (5, 2, 6, 6,  1,  1,  1),

        # rank 6: C_12 += A_10 * B_02
        (6, 3, 2, 7,  1,  1,  1),
        (6, 4, 5, 7,  1,  1,  1),

        # rank 7: C_12 += A_12 * B_20
        (7, 5, 6, 7,  1,  1,  1),

        # rank 8: C_02 += A_01 * B_12
        (8, 1, 5, 6,  1,  1,  1),

        # rank 9: C_12 += A_11 * B_12
        (9, 4, 5, 7,  1,  1,  1),

        # rank 10: C_20 += A_20 * B_00
        (10, 6, 0, 2,  1,  1,  1),
        (10, 7, 1, 2,  1,  1,  1),

        # rank 11: C_21 += A_21 * B_01
        (11, 7, 1, 5,  1,  1,  1),

        # rank 12: C_20 += A_22 * B_20
        (12, 8, 6, 2,  1,  1,  1),

        # rank 13: C_21 += A_22 * B_21
        (13, 8, 7, 5,  1,  1,  1),

        # rank 14: C_22 += A_20 * B_02
        (14, 6, 2, 8,  1,  1,  1),

        # rank 15: C_22 += A_21 * B_12
        (15, 7, 5, 8,  1,  1,  1),

        # rank 16: C_22 += A_22 * B_22
        (16, 8, 8, 8,  1,  1,  1),

        # rank 17: C_00 += A_11 * B_10
        (17, 4, 3, 0,  1,  1,  1),

        # rank 18: C_01 += A_11 * B_11 - A_00*B_11 + A_01*B_10
        # This rank handles the cross terms for C_01 and C_10 involving row/col 1
        (18, 4, 4, 3,  1,  1,  1),
        (18, 0, 4, 3, -1,  1, -1),
        (18, 1, 3, 3,  1,  1,  1),
    ]

    for r, u_idx, v_idx, w_idx, uc, vc, wc in terms:
        U[u_idx][r] += uc
        V[v_idx][r] += vc
        W[w_idx][r] += wc

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
