"""Program for matrix multiplication 2x4 by 4x6 using Strassen decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 multiplication
    # X indices: 0=X00, 1=X01, 2=X10, 3=X11
    # Y indices: 0=Y00, 1=Y01, 2=Y10, 3=Y11
    # Z indices: 0=Z00, 1=Z01, 2=Z10, 3=Z11
    
    # Each term is (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: dict {local_x_idx: val}
    # y_coeffs: dict {local_y_idx: val}
    # z_coeffs: dict {local_z_idx: val}
    
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11) -> Z00(+1), Z11(+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (X10+X11)Y00 -> Z10(+1), Z11(-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = X00(Y01-Y11) -> Z01(+1), Z11(+1)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = X11(Y10-Y00) -> Z00(+1), Z10(+1)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (X00+X01)Y11 -> Z00(-1), Z01(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (X10-X00)(Y00+Y01) -> Z11(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (X01-X11)(Y10+Y11) -> Z00(+1), Z11(+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),
    ]
    
    # Define the 6 blocks: (a_col_off, b_row_off, b_col_off, c_col_off)
    # A is 2x4, so a_row_off is always 0, a spans cols [a_col_off, a_col_off+1]
    # B is 4x6, b spans rows [b_row_off, b_row_off+1], cols [b_col_off, b_col_off+1]
    # C is 2x6, c_row_off is always 0, c spans cols [c_col_off, c_col_off+1]
    
    blocks = [
        (0, 0, 0, 0),  # A1 * B1 -> C1
        (2, 2, 0, 0),  # A2 * B4 -> C1
        (0, 0, 2, 2),  # A1 * B2 -> C2
        (2, 2, 2, 2),  # A2 * B5 -> C2
        (0, 0, 4, 4),  # A1 * B3 -> C3
        (2, 2, 4, 4),  # A2 * B6 -> C3
    ]
    
    for b_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(blocks):
        start_col = b_idx * 7
        for s_idx, (x_coeffs, y_coeffs, z_coeffs) in enumerate(strassen_terms):
            col = start_col + s_idx
            
            # Fill U
            for x_idx, val in x_coeffs.items():
                r = x_idx // 2
                c = x_idx % 2
                global_i = r  # a_row_off is 0
                global_j = a_col_off + c
                U[global_i * m + global_j][col] += val
            
            # Fill V
            for y_idx, val in y_coeffs.items():
                r = y_idx // 2
                c = y_idx % 2
                global_j = b_row_off + r
                global_k = b_col_off + c
                V[global_j * p + global_k][col] += val
            
            # Fill W
            for z_idx, val in z_coeffs.items():
                r = z_idx // 2
                c = z_idx % 2
                global_i = r  # c_row_off is 0
                global_k = c_col_off + c
                W[global_k * n + global_i][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
