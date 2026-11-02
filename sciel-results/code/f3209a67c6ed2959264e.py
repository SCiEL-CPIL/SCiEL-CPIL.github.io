"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_2x2x4(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col_idx
        # a_row, a_col: top-left of 2x2 A block
        # b_row, b_col: top-left of 2x4 B block
        # c_row, c_col: top-left of 2x4 C block (in C space, rows=n, cols=p)
        # W index is k*n + i. k is col of C, i is row of C.
        
        # A indices
        A00 = a_row * m + a_col
        A01 = a_row * m + a_col + 1
        A10 = (a_row + 1) * m + a_col
        A11 = (a_row + 1) * m + a_col + 1
        
        # B indices (2x4)
        # Row 0: b_row, cols b_col..b_col+3
        # Row 1: b_row+1, cols b_col..b_col+3
        B00 = b_row * p + b_col
        B01 = b_row * p + b_col + 1
        B02 = b_row * p + b_col + 2
        B03 = b_row * p + b_col + 3
        B10 = (b_row + 1) * p + b_col
        B11 = (b_row + 1) * p + b_col + 1
        B12 = (b_row + 1) * p + b_col + 2
        B13 = (b_row + 1) * p + b_col + 3
        
        # C indices (2x4) in W (p x n)
        # C row i (0,1), col k (0..3)
        # W index: (c_col + k) * n + (c_row + i)
        
        def add_term(coeffs_U, coeffs_V, coeffs_W):
            nonlocal col_idx
            for idx, val in coeffs_U:
                U[idx][col_idx] += val
            for idx, val in coeffs_V:
                V[idx][col_idx] += val
            for idx, val in coeffs_W:
                W[idx][col_idx] += val
            col_idx += 1
            
        # Strassen for A (2x2) * B_L (2x2)
        # B_L cols: 0, 1
        # B_R cols: 2, 3
        
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(
            [(A00, 1), (A11, 1)],
            [(B00, 1), (B11, 1)],
            [( (c_col + 0) * n + (c_row + 0), 1), ( (c_col + 1) * n + (c_row + 1), -1)] # C00 +1, C11 -1? 
            # Check Strassen formula:
            # C11 = M1 - M4 + M6
            # C12 = M5 + M7
            # C21 = M2 + M4
            # C22 = M1 - M3 + M5
            # So M1 contributes +1 to C11 (C00) and +1 to C22 (C11).
            # My indices: c_row is row of C. c_col is col of C.
            # C00 is row 0, col 0. C11 is row 1, col 1.
            # W index for C00: (c_col+0)*n + (c_row+0)
            # W index for C11: (c_col+1)*n + (c_row+1)
            # So both +1.
        )
        # Correction: M1 contributes to C00 and C11 with +1.
        U[A00][col_idx-1] = 1
        U[A11][col_idx-1] = 1
        V[B00][col_idx-1] = 1
        V[B11][col_idx-1] = 1
        W[(c_col + 0) * n + (c_row + 0)][col_idx-1] = 1
        W[(c_col + 1) * n + (c_row + 1)][col_idx-1] = 1
        
        # M2 = (A10 + A11) * B00
        add_term(
            [(A10, 1), (A11, 1)],
            [(B00, 1)],
            [( (c_col + 0) * n + (c_row + 1), 1)] # C10 (row 1, col 0)
        )
        
        # M3 = A00 * (B10 - B11)
        add_term(
            [(A00, 1)],
            [(B10, 1), (B11, -1)],
            [( (c_col + 1) * n + (c_row + 1), -1)] # C11 (row 1, col 1)
        )
        
        # M4 = A11 * (B01 - B00)
        add_term(
            [(A11, 1)],
            [(B01, 1), (B00, -1)],
            [( (c_col + 0) * n + (c_row + 0), -1), ( (c_col + 0) * n + (c_row + 1), 1)] # C00 -1, C10 +1
        )
        
        # M5 = (A00 + A01) * B11
        add_term(
            [(A00, 1), (A01, 1)],
            [(B11, 1)],
            [( (c_col + 1) * n + (c_row + 0), 1), ( (c_col + 1) * n + (c_row + 1), 1)] # C01 +1, C11 +1
        )
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term(
            [(A10, 1), (A00, -1)],
            [(B00, 1), (B01, 1)],
            [( (c_col + 0) * n + (c_row + 0), 1)] # C00 +1
        )
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term(
            [(A01, 1), (A11, -1)],
            [(B10, 1), (B11, 1)],
            [( (c_col + 1) * n + (c_row + 0), 1)] # C01 +1
        )
        
        # Now for B_R (cols 2, 3)
        # B_R00 = B02, B_R01 = B03
        # B_R10 = B12, B_R11 = B13
        
        # M8 = (A00 + A11) * (B02 + B13)
        add_term(
            [(A00, 1), (A11, 1)],
            [(B02, 1), (B13, 1)],
            [( (c_col + 2) * n + (c_row + 0), 1), ( (c_col + 3) * n + (c_row + 1), 1)]
        )
        
        # M9 = (A10 + A11) * B02
        add_term(
            [(A10, 1), (A11, 1)],
            [(B02, 1)],
            [( (c_col + 2) * n + (c_row + 1), 1)]
        )
        
        # M10 = A00 * (B12 - B13)
        add_term(
            [(A00, 1)],
            [(B12, 1), (B13, -1)],
            [( (c_col + 3) * n + (c_row + 1), -1)]
        )
        
        # M11 = A11 * (B03 - B02)
        add_term(
            [(A11, 1)],
            [(B03, 1), (B02, -1)],
            [( (c_col + 2) * n + (c_row + 0), -1), ( (c_col + 2) * n + (c_row + 1), 1)]
        )
        
        # M12 = (A00 + A01) * B13
        add_term(
            [(A00, 1), (A01, 1)],
            [(B13, 1)],
            [( (c_col + 3) * n + (c_row + 0), 1), ( (c_col + 3) * n + (c_row + 1), 1)]
        )
        
        # M13 = (A10 - A00) * (B02 + B03)
        add_term(
            [(A10, 1), (A00, -1)],
            [(B02, 1), (B03, 1)],
            [( (c_col + 2) * n + (c_row + 0), 1)]
        )
        
        # M14 = (A01 - A11) * (B12 + B13)
        add_term(
            [(A01, 1), (A11, -1)],
            [(B12, 1), (B13, 1)],
            [( (c_col + 3) * n + (c_row + 0), 1)]
        )
        
    def add_schoolbook_1x2x4(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col_idx
        # a_row, a_col: top-left of 1x2 A block
        # b_row, b_col: top-left of 2x4 B block
        # c_row, c_col: top-left of 1x4 C block
        
        A0 = a_row * m + a_col
        A1 = a_row * m + a_col + 1
        
        B00 = b_row * p + b_col
        B01 = b_row * p + b_col + 1
        B02 = b_row * p + b_col + 2
        B03 = b_row * p + b_col + 3
        B10 = (b_row + 1) * p + b_col
        B11 = (b_row + 1) * p + b_col + 1
        B12 = (b_row + 1) * p + b_col + 2
        B13 = (b_row + 1) * p + b_col + 3
        
        # C00 = A0*B00 + A1*B10
        # C01 = A0*B01 + A1*B11
        # ...
        
        for k in range(4):
            # Term 1: A0 * B0k
            add_term(
                [(A0, 1)],
                [(b_row * p + b_col + k, 1)],
                [( (c_col + k) * n + c_row, 1)]
            )
            # Term 2: A1 * B1k
            add_term(
                [(A1, 1)],
                [((b_row + 1) * p + b_col + k, 1)],
                [( (c_col + k) * n + c_row, 1)]
            )
            
    # 1. A00, B1L, C00
    add_strassen_2x2x4(0, 0, 0, 0, 0, 0)
    # 2. A00, B1R, C01
    add_strassen_2x2x4(0, 0, 0, 4, 0, 4)
    # 3. A01, B2L, C00
    add_strassen_2x2x4(0, 2, 2, 0, 0, 0)
    # 4. A01, B2R, C01
    add_strassen_2x2x4(0, 2, 2, 4, 0, 4)
    # 5. A10, B1L, C10
    add_schoolbook_1x2x4(2, 0, 0, 0, 2, 0)
    # 6. A10, B1R, C11
    add_schoolbook_1x2x4(2, 0, 0, 4, 2, 4)
    # 7. A11, B2L, C10
    add_schoolbook_1x2x4(2, 2, 2, 0, 2, 0)
    # 8. A11, B2R, C11
    add_schoolbook_1x2x4(2, 2, 2, 4, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
