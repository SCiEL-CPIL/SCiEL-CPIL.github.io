"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a Strassen-like decomposition for 2x2 blocks.
    # A is 2x4, split into two 2x2 blocks A1 (cols 0,1) and A2 (cols 2,3).
    # B is 4x6, split into six 2x2 blocks:
    # B1 (rows 0,1 cols 0,1), B2 (rows 0,1 cols 2,3), B3 (rows 0,1 cols 4,5)
    # B4 (rows 2,3 cols 0,1), B5 (rows 2,3 cols 2,3), B6 (rows 2,3 cols 4,5)
    # C is 2x6.
    # C1 (cols 0,1) = A1*B1 + A2*B4
    # C2 (cols 2,3) = A1*B2 + A2*B5
    # C3 (cols 4,5) = A1*B3 + A2*B6
    
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # There are 6 such multiplications. Total rank = 42.
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 products definitions for 2x2 matrices X, Y
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Result Z:
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We define the 7 multipliers as tuples of (coeff_X, coeff_Y) for each element
    # X elements: X00, X01, X10, X11 (row-major 2x2)
    # Y elements: Y00, Y01, Y10, Y11
    
    # M1: X00+X11, Y00+Y11
    # M2: X10+X11, Y00
    # M3: X00, Y01-Y11
    # M4: X11, Y10-Y00
    # M5: X00+X01, Y11
    # M6: X10-X00, Y00+Y01
    # M7: X01-X11, Y10+Y11
    
    strassen_defs = [
        # (x_coeffs, y_coeffs) where coeffs are dicts {index: val}
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # Result reconstruction coeffs for Z00, Z01, Z10, Z11 from M1..M7
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    res_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # Z00
        [0, 0, 1, 0, 1, 0, 0],  # Z01
        [0, 1, 0, 1, 0, 0, 0],  # Z10
        [1, -1, 1, 0, 0, 1, 0]  # Z11
    ]
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """
        Adds the Strassen decomposition for a 2x2 block multiplication.
        a_row_offset: starting row in A for the 2x2 block (0 or 1 is not applicable here as A is 2x4, blocks are 2x2 so row offset is always 0 for the 2 rows? No, A is 2x4. A1 is cols 0-1, A2 is cols 2-3. Rows are 0-1. So a_row_offset is always 0 for the 2x2 block in A? Yes, the block spans rows 0-1 of A.
        Wait, A is 2x4. The blocks are 2x2. So they span all 2 rows of A.
        So a_row_offset is always 0.
        a_col_offset: 0 for A1, 2 for A2.
        b_row_offset: 0 for B1,B2,B3 (rows 0-1), 2 for B4,B5,B6 (rows 2-3).
        b_col_offset: 0 for B1,B4, 2 for B2,B5, 4 for B3,B6.
        c_row_offset: 0 (C is 2x6, blocks span rows 0-1).
        c_col_offset: 0 for C1, 2 for C2, 4 for C3.
        """
        for s in range(7):
            col = start_col + s
            x_coeffs, y_coeffs = strassen_defs[s]
            
            # Fill U
            for x_idx, val in x_coeffs.items():
                # x_idx 0,1,2,3 correspond to local (0,0), (0,1), (1,0), (1,1)
                # Global A index: row = a_row_offset + x_idx//2, col = a_col_offset + x_idx%2
                # But A is 2x4. Local 2x2.
                # Local (r,c) -> Global (a_row_offset + r, a_col_offset + c)
                # U index: i*m + j.
                r = x_idx // 2
                c = x_idx % 2
                global_i = a_row_offset + r
                global_j = a_col_offset + c
                U[global_i * m + global_j][col] = val
                
            # Fill V
            for y_idx, val in y_coeffs.items():
                # y_idx 0,1,2,3 correspond to local (0,0), (0,1), (1,0), (1,1)
                # Global B index: row = b_row_offset + y_idx//2, col = b_col_offset + y_idx%2
                r = y_idx // 2
                c = y_idx % 2
                global_j = b_row_offset + r
                global_k = b_col_offset + c
                V[global_j * p + global_k][col] = val
                
            # Fill W
            # The result of this Strassen step is a 2x2 matrix Z.
            # Z contributes to C at c_row_offset, c_col_offset.
            # Z00 -> C[c_row_offset + 0, c_col_offset + 0]
            # Z01 -> C[c_row_offset + 0, c_col_offset + 1]
            # Z10 -> C[c_row_offset + 1, c_col_offset + 0]
            # Z11 -> C[c_row_offset + 1, c_col_offset + 1]
            
            # W index: k*n + i
            for z_idx in range(4):
                # z_idx 0,1,2,3 correspond to Z00, Z01, Z10, Z11
                coeff = res_coeffs[z_idx][s]
                if coeff != 0:
                    z_r = z_idx // 2
                    z_c = z_idx % 2
                    global_i = c_row_offset + z_r
                    global_k = c_col_offset + z_c
                    W[global_k * n + global_i][col] = coeff

    # Block 1: A1 (cols 0-1) * B1 (rows 0-1, cols 0-1) -> C1 (cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A2 (cols 2-3) * B4 (rows 2-3, cols 0-1) -> C1 (cols 0-1)
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    
    # Block 3: A1 (cols 0-1) * B2 (rows 0-1, cols 2-3) -> C2 (cols 2-3)
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    
    # Block 4: A2 (cols 2-3) * B5 (rows 2-3, cols 2-3) -> C2 (cols 2-3)
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # Block 5: A1 (cols 0-1) * B3 (rows 0-1, cols 4-5) -> C3 (cols 4-5)
    add_strassen_block(0, 0, 0, 4, 0, 4, 28)
    
    # Block 6: A2 (cols 2-3) * B6 (rows 2-3, cols 4-5) -> C3 (cols 4-5)
    add_strassen_block(0, 2, 2, 4, 0, 4, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
