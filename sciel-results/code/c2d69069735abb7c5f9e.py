# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    # Rank 36 decomposition:
    # M_2,4,4 decomposed into 4 M_2,2,2 blocks (2 for A1*B1, 2 for A2*B2)
    # M_2,2,2 rank 7 each. Total 4*7 = 28.
    # M_2,4,1 decomposed into schoolbook rank 8.
    # Total rank 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 16 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    # Strassen M_2,2,2 coefficients
    # M_1: A00+A11, B00+B11
    # M_2: A10+A11, B00
    # M_3: A00, B10-B11
    # M_4: A11, B01-B00
    # M_5: A00+A10, B11
    # M_6: A11-A00, B00+B10
    # M_7: A01-A11, B11+B01
    
    # Output C00 = M1 - M4 + M5 - M7
    # Output C01 = M3 + M7
    # Output C10 = M2 + M6
    # Output C11 = M1 - M3 + M4 + M6
    
    # We need to map these to the global indices.
    # Block 1: A1 (rows 0,1 of A; cols 0,1), B1 (rows 0,1 of B; cols 0,1 of B1 which are cols 0,1 of global B)
    # Wait, B is 4x4.
    # A1 is A[0:2, 0:2]. B1 is B[0:2, 0:4]?
    # My split was:
    # A = [A1, A2] where A1 is 2x2, A2 is 2x2.
    # B = [B1; B2] where B1 is 2x4, B2 is 2x4.
    # C = A1 B1 + A2 B2.
    # B1 is 2x4. Split B1 into B1a (2x2) and B1b (2x2).
    # B1a is B[0:2, 0:2]. B1b is B[0:2, 2:4].
    # B2 is 2x4. Split B2 into B2a (2x2) and B2b (2x2).
    # B2a is B[2:4, 0:2]. B2b is B[2:4, 2:4].
    
    # So we have 4 blocks of M_2,2,2:
    # 1. A1 * B1a (A[0:2,0:2] * B[0:2,0:2]) -> contributes to C[0:2, 0:2]
    # 2. A1 * B1b (A[0:2,0:2] * B[0:2,2:4]) -> contributes to C[0:2, 2:4]
    # 3. A2 * B2a (A[0:2,2:4] * B[2:4,0:2]) -> contributes to C[0:2, 0:2]
    # 4. A2 * B2b (A[0:2,2:4] * B[2:4,2:4]) -> contributes to C[0:2, 2:4]
    
    # Note: C is 2x4.
    # C[0:2, 0:2] = A1 B1a + A2 B2a
    # C[0:2, 2:4] = A1 B1b + A2 B2b
    
    # Let's define a helper to add a Strassen block.
    # Block defined by:
    # A_rows (0,1), A_cols (a_start, a_end)
    # B_rows (b_start, b_end), B_cols (c_start, c_end)
    # C_rows (0,1), C_cols (c_start, c_end)
    
    def add_strassen_block(a_col_start, b_row_start, b_col_start, col_offset):
        # col_offset is the starting column index in the global rank
        # This block computes C[0:2, b_col_start:b_col_start+2] contribution from A[0:2, a_col_start:a_col_start+2] * B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        
        # Local indices for A: 0,1 (rows), 0,1 (cols)
        # Local indices for B: 0,1 (rows), 0,1 (cols)
        # Local indices for C: 0,1 (rows), 0,1 (cols)
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        U[(0*a_col_start + 0) * 1 + 0][col_offset + 0] += 1 # A00
        U[(1*a_col_start + 0) * 1 + 0][col_offset + 0] += 1 # A11? No, A11 is row 1, col 1 of local A.
        
        # Let's use explicit indices.
        # A local (r, c) -> Global U index: r * 4 + (a_col_start + c)
        # B local (r, c) -> Global V index: (b_row_start + r) * 4 + (b_col_start + c)
        # C local (r, c) -> Global W index: (b_col_start + c) * 2 + r
        
        # M1: A00+A11, B00+B11
        U[0 * 4 + a_col_start + 0][col_offset + 0] += 1
        U[1 * 4 + a_col_start + 1][col_offset + 0] += 1
        V[(b_row_start + 0) * 4 + b_col_start + 0][col_offset + 0] += 1
        V[(b_row_start + 1) * 4 + b_col_start + 1][col_offset + 0] += 1
        # C00 += 1, C11 += 1
        W[(b_col_start + 0) * 2 + 0][col_offset + 0] += 1
        W[(b_col_start + 1) * 2 + 1][col_offset + 0] += 1
        
        # M2: A10+A11, B00
        U[1 * 4 + a_col_start + 0][col_offset + 1] += 1
        U[1 * 4 + a_col_start + 1][col_offset + 1] += 1
        V[(b_row_start + 0) * 4 + b_col_start + 0][col_offset + 1] += 1
        # C10 += 1
        W[(b_col_start + 0) * 2 + 1][col_offset + 1] += 1
        
        # M3: A00, B10-B11
        U[0 * 4 + a_col_start + 0][col_offset + 2] += 1
        V[(b_row_start + 1) * 4 + b_col_start + 0][col_offset + 2] += 1
        V[(b_row_start + 1) * 4 + b_col_start + 1][col_offset + 2] -= 1
        # C01 += 1
        W[(b_col_start + 1) * 2 + 0][col_offset + 2] += 1
        
        # M4: A11, B01-B00
        U[1 * 4 + a_col_start + 1][col_offset + 3] += 1
        V[(b_row_start + 0) * 4 + b_col_start + 1][col_offset + 3] += 1
        V[(b_row_start + 0) * 4 + b_col_start + 0][col_offset + 3] -= 1
        # C00 -= 1, C11 += 1
        W[(b_col_start + 0) * 2 + 0][col_offset + 3] -= 1
        W[(b_col_start + 1) * 2 + 1][col_offset + 3] += 1
        
        # M5: A00+A10, B11
        U[0 * 4 + a_col_start + 0][col_offset + 4] += 1
        U[1 * 4 + a_col_start + 0][col_offset + 4] += 1
        V[(b_row_start + 1) * 4 + b_col_start + 1][col_offset + 4] += 1
        # C00 += 1
        W[(b_col_start + 0) * 2 + 0][col_offset + 4] += 1
        
        # M6: A11-A00, B00+B10
        U[1 * 4 + a_col_start + 1][col_offset + 5] += 1
        U[0 * 4 + a_col_start + 0][col_offset + 5] -= 1
        V[(b_row_start + 0) * 4 + b_col_start + 0][col_offset + 5] += 1
        V[(b_row_start + 1) * 4 + b_col_start + 0][col_offset + 5] += 1
        # C10 += 1, C11 += 1
        W[(b_col_start + 0) * 2 + 1][col_offset + 5] += 1
        W[(b_col_start + 1) * 2 + 1][col_offset + 5] += 1
        
        # M7: A01-A11, B11+B01
        U[0 * 4 + a_col_start + 1][col_offset + 6] += 1
        U[1 * 4 + a_col_start + 1][col_offset + 6] -= 1
        V[(b_row_start + 1) * 4 + b_col_start + 1][col_offset + 6] += 1
        V[(b_row_start + 0) * 4 + b_col_start + 1][col_offset + 6] += 1
        # C00 -= 1, C01 += 1
        W[(b_col_start + 0) * 2 + 0][col_offset + 6] -= 1
        W[(b_col_start + 1) * 2 + 0][col_offset + 6] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2, 7)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(2, 2, 0, 14)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(2, 2, 2, 21)
    
    # Now handle the 5th column of B (index 4)
    # This is M_2,4,1: A[0:2, 0:4] * B[0:4, 4:5]
    # Schoolbook: 8 terms.
    col_offset = 28
    for i in range(2):
        for j in range(4):
            U[i * 4 + j][col_offset] = 1
            V[j * 4 + 4][col_offset] = 1
            W[4 * 2 + i][col_offset] = 1 # W[k*n+i], k=4
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
