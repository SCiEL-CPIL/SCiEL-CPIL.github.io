import numpy as np
import time

def construct():
    # Deadline guard: 150s limit, use 140s to be safe
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 8

    # We aim for rank 98 (2 * 49)
    # This is achieved by splitting the 4x8 B matrix into two 4x4 blocks
    # and performing two independent 4x4x4 Strassen multiplications.

    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 base matrices (rank 7)
    # U2, V2, W2 are 4x7 matrices representing coefficients for A, B, C blocks
    # Row index in U2/V2/W2: block_row * 2 + block_col (i.e., 0=A00, 1=A01, 2=A10, 3=A11)
    # Col index: strassen_step (0..6)

    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # Standard Strassen coefficients for 2x2 multiplication
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)

    # A coefficients (U2)
    U2[0, 0] = 1; U2[3, 0] = 1   # P0: A00, A11
    U2[2, 1] = 1; U2[3, 1] = 1   # P1: A10, A11
    U2[0, 2] = 1                 # P2: A00
    U2[3, 3] = 1                 # P3: A11
    U2[0, 4] = 1; U2[1, 4] = 1   # P4: A00, A01
    U2[2, 5] = 1; U2[0, 5] = -1  # P5: A10, -A00
    U2[1, 6] = 1; U2[3, 6] = -1  # P6: A01, -A11

    # B coefficients (V2)
    V2[0, 0] = 1; V2[3, 0] = 1   # P0: B00, B11
    V2[0, 1] = 1                 # P1: B00
    V2[1, 2] = 1; V2[3, 2] = -1  # P2: B01, -B11
    V2[2, 3] = 1; V2[0, 3] = -1  # P3: B10, -B00
    V2[3, 4] = 1                 # P4: B11
    V2[0, 5] = 1; V2[1, 5] = 1   # P5: B00, B01
    V2[2, 6] = 1; V2[3, 6] = 1   # P6: B10, B11

    # C coefficients (W2)
    W2[0, 0] = 1; W2[3, 0] = 1   # P0: C00, C11
    W2[2, 1] = 1; W2[3, 1] = -1  # P1: C10, -C11
    W2[1, 2] = 1; W2[3, 2] = 1   # P2: C01, C11
    W2[0, 3] = 1; W2[2, 3] = 1   # P3: C00, C10
    W2[0, 4] = -1; W2[1, 4] = 1  # P4: -C00, C01
    W2[3, 5] = 1                 # P5: C11
    W2[0, 6] = 1                 # P6: C00

    # Build the 4x4x4 decomposition (rank 49) using recursive Strassen
    # The 4x4 matrix is viewed as a 2x2 of 2x2 blocks.
    # Top-level Strassen gives 7 products P_k of (2x2 blocks of A) * (2x2 blocks of B).
    # Each P_k is a 2x2 * 2x2 multiplication, decomposed with inner Strassen (7 steps S_m).
    # Total: 7 * 7 = 49 rank-1 terms per 4x4x4 block.

    # For the 4x8 problem, we have two 4x4x4 blocks:
    #   Block 0: B cols 0-3, C cols 0-3
    #   Block 1: B cols 4-7, C cols 4-7

    # A_specs[p_k]: list of (coeff, block_idx) for A blocks in top-level Strassen step p_k
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11 (where A00 is top-left 2x2 block, etc.)
    A_specs = [
        [(1, 0), (1, 3)],      # P0: A00 + A11
        [(1, 2), (1, 3)],      # P1: A10 + A11
        [(1, 0)],              # P2: A00
        [(1, 3)],              # P3: A11
        [(1, 0), (1, 1)],      # P4: A00 + A01
        [(1, 2), (-1, 0)],     # P5: A10 - A00
        [(1, 1), (-1, 3)],     # P6: A01 - A11
    ]

    # B_specs[p_k]: list of (coeff, block_idx) for B blocks
    B_specs = [
        [(1, 0), (1, 3)],      # P0: B00 + B11
        [(1, 0)],              # P1: B00
        [(1, 1), (-1, 3)],     # P2: B01 - B11
        [(1, 2), (-1, 0)],     # P3: B10 - B00
        [(1, 3)],              # P4: B11
        [(1, 0), (1, 1)],      # P5: B00 + B01
        [(1, 2), (1, 3)],      # P6: B10 + B11
    ]

    # C_specs[p_k]: list of (coeff, c_block_idx) for C blocks
    C_specs = [
        [(1, 0), (1, 3)],      # P0: C00 + C11
        [(1, 2), (-1, 3)],     # P1: C10 - C11
        [(1, 1), (1, 3)],      # P2: C01 + C11
        [(1, 0), (1, 2)],      # P3: C00 + C10
        [(-1, 0), (1, 1)],     # P4: -C00 + C01
        [(1, 3)],              # P5: C11
        [(1, 0)],              # P6: C00
    ]

    for p_k in range(7):
        for s_m in range(7):
            col1 = p_k * 7 + s_m       # Column for block 0 (B cols 0-3)
            col2 = 49 + p_k * 7 + s_m  # Column for block 1 (B cols 4-7)

            # U contribution: same for both blocks (A is shared)
            for coeff_a, block_idx in A_specs[p_k]:
                bi, bj = divmod(block_idx, 2)  # bi = block row, bj = block col
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_a_idx = global_r * 4 + global_c
                        a_eff_idx = r * 2 + c
                        val = coeff_a * U2[a_eff_idx, s_m]
                        if val != 0:
                            U[global_a_idx, col1] += val
                            U[global_a_idx, col2] += val

            # V and W for Block 0 (B cols 0-3, C cols 0-3)
            for coeff_b, block_idx in B_specs[p_k]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_b_idx = global_r * 8 + global_c
                        b_eff_idx = r * 2 + c
                        val = coeff_b * V2[b_eff_idx, s_m]
                        if val != 0:
                            V[global_b_idx, col1] += val

            for coeff_c, c_block_idx in C_specs[p_k]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r  # row index of C
                        global_k = bj * 2 + c  # col index of C
                        global_w_idx = global_k * 4 + global_i  # W[k*n+i]
                        p_eff_idx = r * 2 + c
                        val = coeff_c * W2[p_eff_idx, s_m]
                        if val != 0:
                            W[global_w_idx, col1] += val

            # V and W for Block 1 (B cols 4-7, C cols 4-7)
            for coeff_b, block_idx in B_specs[p_k]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_b_idx = global_r * 8 + (global_c + 4)
                        b_eff_idx = r * 2 + c
                        val = coeff_b * V2[b_eff_idx, s_m]
                        if val != 0:
                            V[global_b_idx, col2] += val

            for coeff_c, c_block_idx in C_specs[p_k]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_k = bj * 2 + c
                        global_w_idx = (global_k + 4) * 4 + global_i
                        p_eff_idx = r * 2 + c
                        val = coeff_c * W2[p_eff_idx, s_m]
                        if val != 0:
                            W[global_w_idx, col2] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
