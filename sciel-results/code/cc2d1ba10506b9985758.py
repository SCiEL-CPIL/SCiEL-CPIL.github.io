"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication
    using a block-Strassen approach.
    
    Partition each 6x6 matrix into 3x3 blocks of 2x2 submatrices.
    Apply Strassen's rank-7 decomposition to each 2x2 block multiplication.
    Total rank: 9 (result blocks) * 3 (sum over j) * 7 (Strassen) = 189.
    """
    n, m, p = DIMENSIONS
    nb = 3  # number of blocks per dimension
    bs = 2  # block size
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # Each term is defined by:
    # x_coeffs: { (local_row, local_col): coeff } for A block
    # y_coeffs: { (local_row, local_col): coeff } for B block
    # c_contrib: { (local_row, local_col): coeff } for C block
    strassen_terms = [
        # s=0: (A00+A11)*(B00+B11) -> C00 += M0, C11 += M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s=1: (A10+A11)*B00 -> C10 += M1, C11 -= M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s=2: A00*(B01-B11) -> C01 += M2, C11 += M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s=3: A11*(B10-B00) -> C00 += M3, C10 += M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s=4: (A00+A01)*B11 -> C00 -= M4, C01 += M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s=5: (A10-A00)*(B00+B01) -> C11 += M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s=6: (A01-A11)*(B10+B11) -> C00 += M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    rank = nb * nb * (nb * 7)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(nb):
        for k_block in range(nb):
            for j_block in range(nb):
                for x_coeffs, y_coeffs, c_contrib in strassen_terms:
                    # Update U
                    for (lr_a, lc_a), coeff_a in x_coeffs.items():
                        row_a = i_block * bs + lr_a
                        col_a = j_block * bs + lc_a
                        U[row_a * m + col_a][col] += coeff_a
                    
                    # Update V
                    for (lr_b, lc_b), coeff_b in y_coeffs.items():
                        row_b = j_block * bs + lr_b
                        col_b = k_block * bs + lc_b
                        V[row_b * p + col_b][col] += coeff_b
                    
                    # Update W
                    for (lr_c, lc_c), coeff_c in c_contrib.items():
                        row_c = i_block * bs + lr_c
                        col_c = k_block * bs + lc_c
                        w_row = col_c * n + row_c
                        W[w_row][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
