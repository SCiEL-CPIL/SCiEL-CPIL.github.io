"""Matrix multiplication 2x4 x 4x6 rank-minimal decomposition.

Strategy:
  - Decompose A (2x4) into two 2x2 blocks A1, A2.
  - Decompose B (4x6) into six 2x2 blocks B1..B6.
  - Use Strassen (rank 7) for each 2x2 x 2x2 multiplication.
  - Total rank = 6 * 7 = 42.
  - C1 = A1*B1 + A2*B4, C2 = A1*B2 + A2*B5, C3 = A1*B3 + A2*B6.
  - W uses the trace convention W[k*n + i, r].
"""

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42

    # Strassen 7-product definitions for 2x2 matrices
    # X elements indexed as: 0=X00, 1=X01, 2=X10, 3=X11
    # Y elements indexed as: 0=Y00, 1=Y01, 2=Y10, 3=Y11
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1 = (X00+X11)(Y00+Y11)
        ({2: 1, 3: 1}, {0: 1}),             # M2 = (X10+X11)Y00
        ({0: 1}, {1: 1, 3: -1}),            # M3 = X00(Y01-Y11)
        ({3: 1}, {2: 1, 0: -1}),            # M4 = X11(Y10-Y00)
        ({0: 1, 1: 1}, {3: 1}),             # M5 = (X00+X01)Y11
        ({2: 1, 0: -1}, {0: 1, 1: 1}),      # M6 = (X10-X00)(Y00+Y01)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),      # M7 = (X01-X11)(Y10+Y11)
    ]

    # Result reconstruction: Z00, Z01, Z10, Z11 from M1..M7
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    res_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # Z00
        [0, 0, 1, 0, 1, 0, 0],    # Z01
        [0, 1, 0, 1, 0, 0, 0],    # Z10
        [1, -1, 1, 0, 0, 1, 0],   # Z11
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off,
                           c_row_off, c_col_off, start_col):
        for s in range(7):
            col = start_col + s
            x_coeffs, y_coeffs = strassen_defs[s]

            for x_idx, val in x_coeffs.items():
                r = x_idx // 2
                c = x_idx % 2
                gi = a_row_off + r
                gj = a_col_off + c
                U[gi * m + gj][col] = val

            for y_idx, val in y_coeffs.items():
                r = y_idx // 2
                c = y_idx % 2
                gj = b_row_off + r
                gk = b_col_off + c
                V[gj * p + gk][col] = val

            for z_idx in range(4):
                coeff = res_coeffs[z_idx][s]
                if coeff != 0:
                    zr = z_idx // 2
                    zc = z_idx % 2
                    gi = c_row_off + zr
                    gk = c_col_off + zc
                    W[gk * n + gi][col] = coeff

    # Six 2x2 block multiplications:
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[0:2, 4:6]
    # B4 = B[2:4, 0:2], B5 = B[2:4, 2:4], B6 = B[2:4, 4:6]
    # C1 = C[0:2, 0:2] = A1*B1 + A2*B4
    # C2 = C[0:2, 2:4] = A1*B2 + A2*B5
    # C3 = C[0:2, 4:6] = A1*B3 + A2*B6

    add_strassen_block(0, 0, 0, 0, 0, 0, 0)    # A1*B1 -> C1
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)    # A2*B4 -> C1
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)   # A1*B2 -> C2
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)   # A2*B5 -> C2
    add_strassen_block(0, 0, 0, 4, 0, 4, 28)   # A1*B3 -> C3
    add_strassen_block(0, 2, 2, 4, 0, 4, 35)   # A2*B6 -> C3

    return {"U": U, "V": V, "W": W}
