"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 9]
R = 32

# Strassen decomposition for 2x2x2 multiplication
# C = A * B where A, B, C are 2x2
# 7 products M1..M7
def get_strassen_vectors():
    """
    Returns u, v, w vectors for Strassen 2x2x2.
    u vectors: length 4 (indices 0..3 for A[0,0], A[0,1], A[1,0], A[1,1])
    v vectors: length 4 (indices 0..3 for B[0,0], B[0,1], B[1,0], B[1,1])
    w vectors: length 4 (indices 0..3 for C[0,0], C[0,1], C[1,0], C[1,1])
    """
    # M1 = A00 * (B01 - B11)
    u1 = [1, 0, 0, 0]
    v1 = [0, 1, 0, -1]
    w1 = [0, 1, 0, 0]  # contributes to C01

    # M2 = (A00 + A10) * B11
    u2 = [1, 0, 1, 0]
    v2 = [0, 0, 0, 1]
    w2 = [0, 0, 1, 1]  # contributes to C10 and C11

    # M3 = (A01 - A11) * (B10 + B11)
    u3 = [0, 1, 0, -1]
    v3 = [0, 0, 1, 1]
    w3 = [0, 0, 0, -1]  # contributes to C11 with -1

    # M4 = A11 * (B00 - B10)
    u4 = [0, 0, 0, 1]
    v4 = [1, 0, -1, 0]
    w4 = [0, 0, 1, 0]  # contributes to C10

    # M5 = (A00 + A01) * B00
    u5 = [1, 1, 0, 0]
    v5 = [1, 0, 0, 0]
    w5 = [1, 0, 0, 0]  # contributes to C00

    # M6 = (A10 - A00) * (B00 + B01)
    u6 = [0, 0, 1, -1]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]  # contributes to C11

    # M7 = (A01 - A10) * (B11 - B10)
    u7 = [0, 1, -1, 0]
    v7 = [0, 0, -1, 1]
    w7 = [1, 0, 0, 0]  # contributes to C00

    return [u1, u2, u3, u4, u5, u6, u7], [v1, v2, v3, v4, v5, v6, v7], [w1, w2, w3, w4, w5, w6, w7]

# Schoolbook decomposition for 2x2x1 multiplication (2x2 A times 2x1 B = 2x1 C)
# 4 products
def get_schoolbook_vectors():
    """
    Returns u, v, w vectors for 2x2x1 multiplication.
    u vectors: length 4 (indices 0..3 for A[0,0], A[0,1], A[1,0], A[1,1])
    v vectors: length 2 (indices 0..1 for B[0], B[1])
    w vectors: length 2 (indices 0..1 for C[0], C[1])
    """
    # M1 = A00 * B0
    u1 = [1, 0, 0, 0]
    v1 = [1, 0]
    w1 = [1, 0]

    # M2 = A01 * B1
    u2 = [0, 1, 0, 0]
    v2 = [0, 1]
    w2 = [1, 0]

    # M3 = A10 * B0
    u3 = [0, 0, 1, 0]
    v3 = [1, 0]
    w3 = [0, 1]

    # M4 = A11 * B1
    u4 = [0, 0, 0, 1]
    v4 = [0, 1]
    w4 = [0, 1]

    return [u1, u2, u3, u4], [v1, v2, v3, v4], [w1, w2, w3, w4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS
    # U: (n*m) x R = 4 x 32
    # V: (m*p) x R = 18 x 32
    # W: (p*n) x R = 18 x 32
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # 4 groups of 2 columns each for Strassen, 1 group of 1 column for schoolbook
    # Groups 0-3: columns 2g, 2g+1 use Strassen (7 products each)
    # Group 4: column 8 uses schoolbook (4 products)
    
    strassen_u, strassen_v, strassen_w = get_strassen_vectors()
    schoolbook_u, schoolbook_v, schoolbook_w = get_schoolbook_vectors()

    for g in range(4):
        # Strassen block for B columns 2g, 2g+1
        base_col = g * 7
        for s in range(7):
            col = base_col + s
            # Fill U
            for i in range(4):
                U[i][col] = strassen_u[s][i]
            # Fill V: rows correspond to B[2g, :] and B[2g+1, :]
            # B has m=2 rows, p=9 columns
            # V row index = j*p + k where j is row of B, k is column of B
            # For Strassen, we use B columns 2g and 2g+1
            for j in range(2):
                for k in range(2):
                    v_idx = j * p + (2 * g + k)
                    V[v_idx][col] = strassen_v[s][j * 2 + k]
            # Fill W: rows correspond to C[?, 2g] and C[?, 2g+1]
            # W index = k*n + i where k is column of C, i is row of C
            for i in range(2):
                for k in range(2):
                    w_idx = (2 * g + k) * n + i
                    W[w_idx][col] = strassen_w[s][k * 2 + i]

    # Group 4: column 8 uses schoolbook
    base_col = 28
    for s in range(4):
        col = base_col + s
        # Fill U
        for i in range(4):
            U[i][col] = schoolbook_u[s][i]
        # Fill V: row corresponds to B column 8
        for j in range(2):
            v_idx = j * p + 8
            V[v_idx][col] = schoolbook_v[s][j]
        # Fill W: rows correspond to C column 8
        for i in range(2):
            w_idx = 8 * n + i
            W[w_idx][col] = schoolbook_w[s][i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
