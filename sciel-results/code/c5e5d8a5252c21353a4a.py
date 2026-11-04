import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 77

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition steps
    # Each step is (alpha, beta, gamma) where each is a list of (i, j, coeff)
    # (i, j) are block indices within the 2x2 block
    strassen_steps = [
        # Step 0: M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # Step 1: M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (1,1,-1))]),
        # Step 2: M2 = A00(B01-B11) -> C01 += M2, C11 += M2
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1), (1,1,1))]),
        # Step 3: M3 = A11(B10-B00) -> C00 += M3, C10 += M3
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),
        # Step 4: M4 = (A00+A01)B11 -> C00 -= M4, C01 += M4
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (0,1,1))]),
        # Step 5: M5 = (A10-A00)(B00+B01) -> C11 += M5
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
        # Step 6: M6 = (A01-A11)(B10+B11) -> C00 += M6
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))])
    ]

    # Part 1: 4x4x4 multiplication (columns 0-48)
    # Using recursive Strassen: 7 outer steps x 7 inner steps = 49 rank
    for s_outer in range(7):
        alpha_outer, beta_outer, gamma_outer = strassen_steps[s_outer]
        for s_inner in range(7):
            alpha_inner, beta_inner, gamma_inner = strassen_steps[s_inner]
            col = s_outer * 7 + s_inner

            # Process U (A matrix, 4x4)
            for (bi, bj, val_o) in alpha_outer:
                for (si, sj, val_i) in alpha_inner:
                    # Global row/col in A
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_o * val_i

            # Process V (B matrix, first 4 columns)
            for (bi, bj, val_o) in beta_outer:
                for (si, sj, val_i) in beta_inner:
                    # Global row/col in B (first 4 cols)
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_o * val_i

            # Process W (C matrix, first 4 columns)
            for (bi, bj, val_o) in gamma_outer:
                for (si, sj, val_i) in gamma_inner:
                    # W index: col * n + row
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_o * val_i

    # Part 2: 4x4x2 multiplication (columns 49-76)
    # A is 4x4, B is 4x2, C is 4x2
    # Partition A into 2x2 blocks
    # B and C are split into top (rows 0-1) and bottom (rows 2-3) 2x2 blocks
    # 4 block multiplications, each 2x2x2 with rank 7, total 28 rank

    # Define the 4 block multiplications
    # (A_block, B_block_row, C_block_row)
    # A_block: (bi, bj) in {0,1}x{0,1}
    # B_block_row: 0 or 1 (top or bottom)
    # C_block_row: 0 or 1 (top or bottom)
    # C0 = A00*B0 + A01*B1 (top part)
    # C1 = A10*B0 + A11*B1 (bottom part)

    mults = [
        ((0, 0), 0, 0),  # A00 * B_top -> C_top
        ((0, 1), 1, 0),  # A01 * B_bot -> C_top
        ((1, 0), 0, 1),  # A10 * B_top -> C_bot
        ((1, 1), 1, 1),  # A11 * B_bot -> C_bot
    ]

    col_start_2 = 49
    for m_idx, (a_block, b_row, c_row) in enumerate(mults):
        a_bi, a_bj = a_block
        for s_inner in range(7):
            alpha_inner, beta_inner, gamma_inner = strassen_steps[s_inner]
            col = col_start_2 + m_idx * 7 + s_inner

            # Process U (A matrix)
            for (si, sj, val_i) in alpha_inner:
                r = (a_bi * 2 + si) * m + (a_bj * 2 + sj)
                U[r][col] += val_i

            # Process V (B matrix, columns 4-5)
            for (si, sj, val_i) in beta_inner:
                r = (b_row * 2 + si) * p + (4 + sj)
                V[r][col] += val_i

            # Process W (C matrix, columns 4-5)
            for (si, sj, val_i) in gamma_inner:
                r = (4 + sj) * n + (c_row * 2 + si)
                W[r][col] += val_i

    return {"U": U, "V": V, "W": W}
