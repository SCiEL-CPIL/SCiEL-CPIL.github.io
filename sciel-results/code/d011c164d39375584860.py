"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Helper to set Strassen block for column pair (2k, 2k+1)
    # A = [[a, b], [c, d]] (indices 0,1,2,3 in U)
    # B columns k0, k1: B = [[b0k0, b0k1], [b1k0, b1k1]]
    # V indices: row0 = B[0, col], row1 = B[1, col]
    #   So V[col] = B[0, col], V[p+col] = B[1, col]
    # W indices: W[k*n + r] for C[r, k]
    #   C is 2x15, so C[r, k] -> W[k*2 + r]
    #   C[0, k] -> W[2k], C[1, k] -> W[2k+1]

    # Strassen for 2x2 * 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # where:
    # M1 = (a + d)(b0k0 + b1k1)
    # M2 = (c + d) * b0k0
    # M3 = a * (b0k1 - b1k1)
    # M4 = d * (b1k0 - b0k0)
    # M5 = (a + b) * b1k1
    # M6 = (c - a) * (b0k0 + b0k1)
    # M7 = (b - d) * (b1k0 + b1k1)

    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # M1 = (a + d)(B[0,k0] + B[1,k1])
        U[0][r1] = 1
        U[3][r1] = 1
        V[k0][r1] = 1
        V[p + k1][r1] = 1

        # M2 = (c + d) * B[0,k0]
        U[2][r2] = 1
        U[3][r2] = 1
        V[k0][r2] = 1

        # M3 = a * (B[0,k1] - B[1,k1])
        U[0][r3] = 1
        V[k1][r3] = 1
        V[p + k1][r3] = -1

        # M4 = d * (B[1,k0] - B[0,k0])
        U[3][r4] = 1
        V[p + k0][r4] = 1
        V[k0][r4] = -1

        # M5 = (a + b) * B[1,k1]
        U[0][r5] = 1
        U[1][r5] = 1
        V[p + k1][r5] = 1

        # M6 = (c - a) * (B[0,k0] + B[0,k1])
        U[2][r6] = 1
        U[0][r6] = -1
        V[k0][r6] = 1
        V[k1][r6] = 1

        # M7 = (b - d) * (B[1,k0] + B[1,k1])
        U[1][r7] = 1
        U[3][r7] = -1
        V[p + k0][r7] = 1
        V[p + k1][r7] = 1

        # C[0, k0] = M1 + M4 - M5 + M7
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1

        # C[1, k0] = M2 + M4
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1

        # C[0, k1] = M3 + M5
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1

        # C[1, k1] = M1 - M2 + M3 + M6
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1

    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    # Trivial block for column 14
    # C[0, 14] = a*B[0,14] + b*B[1,14]
    # C[1, 14] = c*B[0,14] + d*B[1,14]
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3

    # a * B[0,14]
    U[0][r1] = 1
    V[14][r1] = 1
    W[28][r1] = 1

    # b * B[1,14]
    U[1][r2] = 1
    V[p + 14][r2] = 1
    W[28][r2] = 1

    # c * B[0,14]
    U[2][r3] = 1
    V[14][r3] = 1
    W[29][r3] = 1

    # d * B[1,14]
    U[3][r4] = 1
    V[p + 14][r4] = 1
    W[29][r4] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
