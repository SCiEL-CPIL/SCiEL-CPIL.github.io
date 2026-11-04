import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Strategy: split the 3x9 matrix B into three 3x3 blocks.
    # For each block, compute C_i = A * B_i where A is 2x3 and B_i is 3x3.
    # Use a rank-17 decomposition for each 2x3 by 3x3 multiplication.
    # Total rank = 3 * 17 = 51.

    # The rank-17 decomposition for A (2x3) * B_i (3x3) -> C_i (2x3):
    # Split A into A_L (2x2, cols 0,1) and A_R (2x1, col 2).
    # Split B_i into B_i_T (2x3, rows 0,1) and B_i_B (1x3, row 2).
    # C_i = A_L * B_i_T + A_R * B_i_B
    #
    # A_L * B_i_T: A_L is 2x2, B_i_T is 2x3.
    #   Split B_i_T into B_i_T1 (2x2, cols 0,1) and B_i_T2 (2x1, col 2).
    #   A_L * B_i_T1: Use Strassen (rank 7).
    #   A_L * B_i_T2: Use outer product (rank 4).
    #   Total: 7 + 4 = 11.
    #
    # A_R * B_i_B: A_R is 2x1, B_i_B is 1x3.
    #   Use outer product (rank 6).
    #
    # Total per block: 11 + 6 = 17.

    rank = 51

    U = [[0] * rank for _ in range(n * m)]  # 6 x 51
    V = [[0] * rank for _ in range(m * p)]  # 27 x 51
    W = [[0] * rank for _ in range(p * n)]  # 18 x 51

    col = 0
    for block in range(3):
        # Block offset in B (columns)
        b_col_offset = block * 3
        # Block offset in C (columns)
        c_col_offset = block * 3

        # Strassen terms for A_L (2x2) * B_i_T1 (2x2) -> C_i_T1 (2x2)
        # A_L = [[A00, A01], [A10, A11]] (rows 0,1 of A; cols 0,1)
        # B_i_T1 = [[B00, B01], [B10, B11]] (rows 0,1 of B_i; cols 0,1)
        # C_i_T1 = [[C00, C01], [C10, C11]] (rows 0,1 of C_i; cols 0,1)
        #
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        for s in range(7):
            if s == 0:
                # M1 = (A00+A11)(B00+B11)
                U[0][col] = 1  # A00
                U[3][col] = 1  # A11
                V[b_col_offset][col] = 1      # B00
                V[b_col_offset + 4][col] = 1   # B11
                W[c_col_offset * n + 0][col] = 1     # C00
                W[(c_col_offset + 1) * n + 0][col] = 0
                W[c_col_offset * n + 1][col] = 0
                W[(c_col_offset + 1) * n + 1][col] = 1     # C11
            elif s == 1:
                # M2 = (A10+A11)B00
                U[3][col] = 1  # A11
                U[1][col] = 1  # A10 (row 1, col 0 of A)
                V[b_col_offset][col] = 1      # B00
                W[c_col_offset * n + 1][col] = 1     # C10
            elif s == 2:
                # M3 = A00(B01-B11)
                U[0][col] = 1  # A00
                V[b_col_offset + 1][col] = 1   # B01
                V[b_col_offset + 4][col] = -1  # -B11
                W[c_col_offset * n + 0][col] = 1     # C00
                W[(c_col_offset + 1) * n + 1][col] = 1     # C11
            elif s == 3:
                # M4 = A11(B10-B00)
                U[3][col] = 1  # A11
                V[b_col_offset + 3][col] = 1   # B10
                V[b_col_offset][col] = -1      # -B00
                W[c_col_offset * n + 0][col] = 1     # C00
                W[c_col_offset * n + 1][col] = 1     # C10
            elif s == 4:
                # M5 = (A00+A01)B11
                U[0][col] = 1  # A00
                U[1][col] = 1  # A01 (row 0, col 1 of A)
                V[b_col_offset + 4][col] = 1   # B11
                W[c_col_offset * n + 0][col] = -1     # -C00
                W[(c_col_offset + 1) * n + 1][col] = 1     # C11
            elif s == 5:
                # M6 = (A10-A00)(B00+B01)
                U[1][col] = 1  # A10
                U[0][col] = -1 # -A00
                V[b_col_offset][col] = 1      # B00
                V[b_col_offset + 1][col] = 1  # B01
                W[(c_col_offset + 1) * n + 1][col] = 1     # C11
            elif s == 6:
                # M7 = (A01-A11)(B10+B11)
                U[1][col] = 1  # A01
                U[3][col] = -1 # -A11
                V[b_col_offset + 3][col] = 1   # B10
                V[b_col_offset + 4][col] = 1   # B11
                W[c_col_offset * n + 0][col] = 1     # C00
            col += 1

        # Outer product terms for A_L (2x2) * B_i_T2 (2x1) -> C_i_T2 (2x1)
        # A_L = [[A00, A01], [A10, A11]]
        # B_i_T2 = [B02, B12] (rows 0,1 of B_i; col 2)
        # C_i_T2 = [C02, C12] (rows 0,1 of C_i; col 2)
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12

        # Term 8: A00*B02
        U[0][col] = 1  # A00
        V[b_col_offset + 2][col] = 1  # B02
        W[(c_col_offset + 2) * n + 0][col] = 1  # C02
        col += 1

        # Term 9: A01*B12
        U[1][col] = 1  # A01
        V[b_col_offset + 5][col] = 1  # B12
        W[(c_col_offset + 2) * n + 0][col] = 1  # C02
        col += 1

        # Term 10: A10*B02
        U[1][col] = 1  # A10
        V[b_col_offset + 2][col] = 1  # B02
        W[(c_col_offset + 2) * n + 1][col] = 1  # C12
        col += 1

        # Term 11: A11*B12
        U[3][col] = 1  # A11
        V[b_col_offset + 5][col] = 1  # B12
        W[(c_col_offset + 2) * n + 1][col] = 1  # C12
        col += 1

        # Outer product terms for A_R (2x1) * B_i_B (1x3) -> C_i_B (2x3)
        # A_R = [A02, A12] (col 2 of A)
        # B_i_B = [B20, B21, B22] (row 2 of B_i)
        # C_i_B = [C00, C01, C02] for row 0, [C10, C11, C12] for row 1
        # C00 += A02*B20
        # C01 += A02*B21
        # C02 += A02*B22
        # C10 += A12*B20
        # C11 += A12*B21
        # C12 += A12*B22

        for k in range(3):
            # Term: A02 * B2k
            U[2][col] = 1  # A02 (row 0, col 2 of A)
            V[b_col_offset + 6 + k][col] = 1  # B2k (row 2, col k of B_i)
            W[(c_col_offset + k) * n + 0][col] = 1  # C0k
            col += 1

            # Term: A12 * B2k
            U[5][col] = 1  # A12 (row 1, col 2 of A)
            V[b_col_offset + 6 + k][col] = 1  # B2k (row 2, col k of B_i)
            W[(c_col_offset + k) * n + 1][col] = 1  # C1k
            col += 1

    return {"U": U, "V": V, "W": W}

