"""Matrix multiplication 2x3 x 3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=4

    # Schoolbook fallback (rank 24)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1

    # Try rank 22 via Strassen on 2x2 blocks + direct 2x1*1x4
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))

    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val

    def add_strassen_block(a_rows, b_rows, c_cols, offset):
        # a_rows: [(i,j) for (0,0),(0,1),(1,0),(1,1)]
        # b_rows: [(j,k) for (0,0),(0,1),(1,0),(1,1)]
        # c_cols: [(k,i) for C00, C11, C10, C01]
        # m1: (a00+a11)(b00+b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 0, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 0, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 0, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 0, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 0, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 0, 1)
        # m2: (a10+a11)b00
        add_u(a_rows[2][0], a_rows[2][1], offset + 1, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 1, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 1, 1)
        add_w(c_cols[2][0], c_cols[2][1], offset + 1, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 1, -1)
        # m3: a00(b01-b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 2, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 2, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 2, -1)
        add_w(c_cols[3][0], c_cols[3][1], offset + 2, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 2, 1)
        # m4: a11(b10-b00)
        add_u(a_rows[3][0], a_rows[3][1], offset + 3, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 3, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 3, -1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 3, 1)
        add_w(c_cols[2][0], c_cols[2][1], offset + 3, 1)
        # m5: (a00+a01)b11
        add_u(a_rows[0][0], a_rows[0][1], offset + 4, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 4, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 4, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 4, -1)
        add_w(c_cols[3][0], c_cols[3][1], offset + 4, 1)
        # m6: (a10-a00)(b00+b10)
        add_u(a_rows[2][0], a_rows[2][1], offset + 5, 1)
        add_u(a_rows[0][0], a_rows[0][1], offset + 5, -1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 5, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 5, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 5, 1)
        # m7: (a01-a11)(b10+b11)
        add_u(a_rows[1][0], a_rows[1][1], offset + 6, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 6, -1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 6, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 6, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 6, 1)

    # Block 1: A[0:2,0:2] * B[0:2,0:2]
    a_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    b_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    c_cols_1 = [(0,0), (1,1), (0,1), (1,0)]
    add_strassen_block(a_rows_1, b_rows_1, c_cols_1, 0)

    # Block 2: A[0:2,0:2] * B[0:2,2:4]
    b_rows_2 = [(0,2), (0,3), (1,2), (1,3)]
    c_cols_2 = [(2,0), (3,1), (2,1), (3,0)]
    add_strassen_block(a_rows_1, b_rows_2, c_cols_2, 7)

    # Block 3: A[0:2,2] * B[2,0:4] (rank 8)
    offset = 14
    for i in range(2):
        for k in range(4):
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1

    # Verify
    T = np.zeros((p, n, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k, i, j] = 1

    valid = True
    for k in range(p):
        for i in range(n):
            for j in range(m):
                total = 0
                for r in range(rank_22):
                    total += U22[i*m+j, r] * V22[j*p+k, r] * W22[k*n+i, r]
                if not np.isclose(total, T[k, i, j]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
