import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)

    # Block-Strassen decomposition for 2x4 * 4x8
    # Split A (2x4) into A_left (2x2) and A_right (2x2)
    # Split B (4x8) into B_top (2x8) and B_bottom (2x8)
    # C = A_left * B_top + A_right * B_bottom
    # Each term: 2x2 * 2x8 = 4 independent 2x2 * 2x2 multiplications
    # Each 2x2 * 2x2 uses Strassen (rank 7)
    # Total rank = 2 * 4 * 7 = 56

    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen for 2x2:
    # A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]], C = A*B
    # M1=(a0+a3)(b0+b3), M2=(a2+a3)b0, M3=a0(b1-b3),
    # M4=a3(b2-b0), M5=(a0+a1)b3, M6=(a2-a0)(b0+b1),
    # M7=(a1-a3)(b2+b3)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6

    u_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v_coeffs = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # 8 blocks: t=0..7
    # t<4: A_left (cols 0,1), B_top (rows 0,1)
    # t>=4: A_right (cols 2,3), B_bottom (rows 2,3)
    # b_col_offset = (t%4)*2

    for t in range(8):
        if t < 4:
            a_col_off = 0
            b_row_off = 0
        else:
            a_col_off = 2
            b_row_off = 2
        b_col_off = (t % 4) * 2

        a_idx = [a_col_off, a_col_off + 1, m + a_col_off, m + a_col_off + 1]
        b_idx = [b_row_off * p + b_col_off, b_row_off * p + b_col_off + 1,
                 (b_row_off + 1) * p + b_col_off, (b_row_off + 1) * p + b_col_off + 1]
        c_idx = [b_col_off * n, (b_col_off + 1) * n, b_col_off * n + 1, (b_col_off + 1) * n + 1]

        for l in range(7):
            r = t * 7 + l
            for li, c in enumerate(u_coeffs[l]):
                if c:
                    U[a_idx[li], r] += c
            for li, c in enumerate(v_coeffs[l]):
                if c:
                    V[b_idx[li], r] += c
            for li, c in enumerate(w_coeffs[l]):
                if c:
                    W[c_idx[li], r] += c

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
