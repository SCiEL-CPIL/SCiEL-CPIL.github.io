"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 80
    rank = 80
    
    U = [[0] * rank for _ in range(n * m)] # 12 x 80
    V = [[0] * rank for _ in range(m * p)] # 32 x 80
    W = [[0] * rank for _ in range(p * n)] # 24 x 80
    
    # Helper to add a 2x2x4 block (rank 12)
    # A indices: (a_r0, a_c0), (a_r1, a_c1)
    # B indices: (b_r0, b_c0), (b_r1, b_c1)
    # C indices: (c_r0, c_c0), (c_r1, c_c1)
    # start_col: starting column in U, V, W
    def add_2x2x4(a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1, c_r0, c_c0, c_r1, c_c1, start_col):
        # m1 = a00 * b00
        U[a_r0 * m + a_c0][start_col] = 1
        V[b_r0 * p + b_c0][start_col] = 1
        W[c_r0 * n + c_c0][start_col] = 1
        W[c_r1 * n + c_c1][start_col] = 1
        # m2 = a01 * b10
        U[a_r0 * m + a_c1][start_col+1] = 1
        V[b_r1 * p + b_c0][start_col+1] = 1
        W[c_r0 * n + c_c0][start_col+1] = 1
        W[c_r1 * n + c_c1][start_col+1] = 1
        # m3 = a00 * b01
        U[a_r0 * m + a_c0][start_col+2] = 1
        V[b_r0 * p + b_c1][start_col+2] = 1
        W[c_r0 * n + c_c0+1][start_col+2] = 1
        W[c_r1 * n + c_c1+1][start_col+2] = 1
        # m4 = a01 * b11
        U[a_r0 * m + a_c1][start_col+3] = 1
        V[b_r1 * p + b_c1][start_col+3] = 1
        W[c_r0 * n + c_c0+1][start_col+3] = 1
        W[c_r1 * n + c_c1+1][start_col+3] = 1
        # m5 = a00 * b02
        U[a_r0 * m + a_c0][start_col+4] = 1
        V[b_r0 * p + b_c2][start_col+4] = 1
        W[c_r0 * n + c_c0+2][start_col+4] = 1
        W[c_r1 * n + c_c1+2][start_col+4] = 1
        # m6 = a01 * b12
        U[a_r0 * m + a_c1][start_col+5] = 1
        V[b_r1 * p + b_c2][start_col+5] = 1
        W[c_r0 * n + c_c0+2][start_col+5] = 1
        W[c_r1 * n + c_c1+2][start_col+5] = 1
        # m7 = a00 * b03
        U[a_r0 * m + a_c0][start_col+6] = 1
        V[b_r0 * p + b_c3][start_col+6] = 1
        W[c_r0 * n + c_c0+3][start_col+6] = 1
        W[c_r1 * n + c_c1+3][start_col+6] = 1
        # m8 = a01 * b13
        U[a_r0 * m + a_c1][start_col+7] = 1
        V[b_r1 * p + b_c3][start_col+7] = 1
        W[c_r0 * n + c_c0+3][start_col+7] = 1
        W[c_r1 * n + c_c1+3][start_col+7] = 1
        # m9 = (a10+a11) * (b00+b10)
        U[a_r1 * m + a_c0][start_col+8] = 1
        U[a_r1 * m + a_c1][start_col+8] = 1
        V[b_r0 * p + b_c0][start_col+8] = 1
        V[b_r1 * p + b_c0][start_col+8] = 1
        W[c_r1 * n + c_c0][start_col+8] = 1
        W[c_r0 * n + c_c0][start_col+8] = -1
        W[c_r1 * n + c_c1][start_col+8] = -1
        W[c_r0 * n + c_c1][start_col+8] = -1
        # m10 = (a10+a11) * (b01+b11)
        U[a_r1 * m + a_c0][start_col+9] = 1
        U[a_r1 * m + a_c1][start_col+9] = 1
        V[b_r0 * p + b_c1][start_col+9] = 1
        V[b_r1 * p + b_c1][start_col+9] = 1
        W[c_r1 * n + c_c0+1][start_col+9] = 1
        W[c_r0 * n + c_c0+1][start_col+9] = -1
        W[c_r1 * n + c_c1+1][start_col+9] = -1
        W[c_r0 * n + c_c1+1][start_col+9] = -1
        # m11 = (a10+a11) * (b02+b12)
        U[a_r1 * m + a_c0][start_col+10] = 1
        U[a_r1 * m + a_c1][start_col+10] = 1
        V[b_r0 * p + b_c2][start_col+10] = 1
        V[b_r1 * p + b_c2][start_col+10] = 1
        W[c_r1 * n + c_c0+2][start_col+10] = 1
        W[c_r0 * n + c_c0+2][start_col+10] = -1
        W[c_r1 * n + c_c1+2][start_col+10] = -1
        W[c_r0 * n + c_c1+2][start_col+10] = -1
        # m12 = (a10+a11) * (b03+b13)
        U[a_r1 * m + a_c0][start_col+11] = 1
        U[a_r1 * m + a_c1][start_col+11] = 1
        V[b_r0 * p + b_c3][start_col+11] = 1
        V[b_r1 * p + b_c3][start_col+11] = 1
        W[c_r1 * n + c_c0+3][start_col+11] = 1
        W[c_r0 * n + c_c0+3][start_col+11] = -1
        W[c_r1 * n + c_c1+3][start_col+11] = -1
        W[c_r0 * n + c_c1+3][start_col+11] = -1

    # Helper to add a 1x2x4 block (rank 8)
    def add_1x2x4(a_r0, a_c0, a_c1, b_r0, b_c0, b_r1, b_c1, c_r0, c_c0, start_col):
        # m1 = a00 * b00
        U[a_r0 * m + a_c0][start_col] = 1
        V[b_r0 * p + b_c0][start_col] = 1
        W[c_r0 * n + c_c0][start_col] = 1
        # m2 = a01 * b10
        U[a_r0 * m + a_c1][start_col+1] = 1
        V[b_r1 * p + b_c0][start_col+1] = 1
        W[c_r0 * n + c_c0][start_col+1] = 1
        # m3 = a00 * b01
        U[a_r0 * m + a_c0][start_col+2] = 1
        V[b_r0 * p + b_c1][start_col+2] = 1
        W[c_r0 * n + c_c0+1][start_col+2] = 1
        # m4 = a01 * b11
        U[a_r0 * m + a_c1][start_col+3] = 1
        V[b_r1 * p + b_c1][start_col+3] = 1
        W[c_r0 * n + c_c0+1][start_col+3] = 1
        # m5 = a00 * b02
        U[a_r0 * m + a_c0][start_col+4] = 1
        V[b_r0 * p + b_c2][start_col+4] = 1
        W[c_r0 * n + c_c0+2][start_col+4] = 1
        # m6 = a01 * b12
        U[a_r0 * m + a_c1][start_col+5] = 1
        V[b_r1 * p + b_c2][start_col+5] = 1
        W[c_r0 * n + c_c0+2][start_col+5] = 1
        # m7 = a00 * b03
        U[a_r0 * m + a_c0][start_col+6] = 1
        V[b_r0 * p + b_c3][start_col+6] = 1
        W[c_r0 * n + c_c0+3][start_col+6] = 1
        # m8 = a01 * b13
        U[a_r0 * m + a_c1][start_col+7] = 1
        V[b_r1 * p + b_c3][start_col+7] = 1
        W[c_r0 * n + c_c0+3][start_col+7] = 1

    # Block 1: A[0:3, 0:2] * B[0:2, 0:4] -> C[0:3, 0:4]
    # Top 2x2x4: A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]
    add_2x2x4(0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0)
    # Bottom 1x2x4: A[2:3, 0:2] * B[0:2, 0:4] -> C[2:3, 0:4]
    add_1x2x4(2, 0, 1, 0, 0, 1, 0, 2, 0, 12)

    # Block 2: A[0:3, 0:2] * B[0:2, 4:8] -> C[0:3, 4:8]
    # Top 2x2x4: A[0:2, 0:2] * B[0:2, 4:8] -> C[0:2, 4:8]
    add_2x2x4(0, 0, 1, 1, 0, 4, 1, 4, 0, 4, 1, 5, 20)
    # Bottom 1x2x4: A[2:3, 0:2] * B[0:2, 4:8] -> C[2:3, 4:8]
    add_1x2x4(2, 0, 1, 0, 4, 1, 4, 2, 4, 32)

    # Block 3: A[0:3, 2:4] * B[2:4, 0:4] -> C[0:3, 0:4] (Add to Block 1)
    # Top 2x2x4: A[0:2, 2:4] * B[2:4, 0:4] -> C[0:2, 0:4]
    add_2x2x4(0, 2, 1, 3, 2, 0, 3, 0, 0, 0, 1, 1, 40)
    # Bottom 1x2x4: A[2:3, 2:4] * B[2:4, 0:4] -> C[2:3, 0:4]
    add_1x2x4(2, 2, 3, 2, 0, 3, 0, 2, 0, 52)

    # Block 4: A[0:3, 2:4] * B[2:4, 4:8] -> C[0:3, 4:8] (Add to Block 2)
    # Top 2x2x4: A[0:2, 2:4] * B[2:4, 4:8] -> C[0:2, 4:8]
    add_2x2x4(0, 2, 1, 3, 2, 4, 3, 4, 0, 4, 1, 5, 60)
    # Bottom 1x2x4: A[2:3, 2:4] * B[2:4, 4:8] -> C[2:3, 4:8]
    add_1x2x4(2, 2, 3, 2, 4, 3, 4, 2, 4, 72)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
