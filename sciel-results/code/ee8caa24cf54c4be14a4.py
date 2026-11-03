"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based block decomposition for 2x4 * 4x6
    # Split A (2x4) into top and bottom 2x2 blocks
    # Split B (4x6) into top and bottom 2x6 blocks
    # Use Strassen for each 2x2 * 2x6 block (treating the 2x6 as three 2x2 blocks)
    # Total: 2 (top/bottom) * 3 (three 2x2 column blocks) * 7 (Strassen) = 42
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 2x2 * 2x2 block, the 7 multiplications map to:
    # U: A block rows (2 rows out of 4 total A rows)
    # V: B block rows (2 rows out of 6 total B rows)  
    # W: C block rows (2 rows out of 6 total C rows)

    col = 0
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x6: rows 0,1,2,3; cols 0,1,2,3,4,5
    # C is 2x6: rows 0,1; cols 0,1,2,3,4,5
    # C[i,k] = sum_j A[i,j]*B[j,k]

    # A_top: rows 0,1 of A -> A[0,0..1], A[0,2..3], A[1,0..1], A[1,2..3]
    # A_bottom: rows 2,3 of A -> same as A_top since n=2, m=4
    # Wait, A is 2x4, so it has 2 rows. The "split" is along columns.
    # Actually for Strassen we need 2x2 blocks. A is 2x4, so split into two 2x2 blocks horizontally.
    # B is 4x6, split into 2x2 blocks: 2 row blocks x 3 col blocks.
    
    # Let me reconsider. A (2x4) = [A_left(2x2) | A_right(2x2)]
    # B (4x6) = [B_top(2x6); B_bottom(2x6)] where B_top is rows 0-1, B_bottom is rows 2-3
    # B_top = [B_t1(2x2) | B_t2(2x2) | B_t3(2x2)]
    # B_bottom = [B_b1(2x2) | B_b2(2x2) | B_b3(2x2)]
    
    # C = A_left * B_top_left_half + A_right * B_top_right_half... no.
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # = (A_left[i,:] * B_top[:,k]) + (A_right[i,:] * B_bottom[:,k])
    
    # So C = A_left * B_top + A_right * B_bottom
    # where A_left is 2x2 (cols 0,1 of A), A_right is 2x2 (cols 2,3 of A)
    # B_top is 2x6 (rows 0,1 of B), B_bottom is 2x6 (rows 2,3 of B)
    
    # Each 2x2 * 2x6 product can be done with 3 Strassen 2x2*2x2 operations (one per 2-col block of the 2x6)
    # So total: 2 * 3 * 7 = 42 multiplications.

    def add_strassen_block(A_row_offset, B_row_offset, C_col_offset, start_col):
        """Add Strassen 2x2*2x2 block contributions.
        A_row_offset: which 2 rows of A (0 for left block cols 0-1, 1 for right block cols 2-3)
        B_row_offset: which 2 rows of B (0 for top rows 0-1, 1 for bottom rows 2-3)  
        C_col_offset: which 2 columns of C (0,1,2 for the three 2-col blocks)
        start_col: starting column in U,V,W
        """
        global col
        a_col_base = A_row_offset * 2  # 0 or 2
        b_row_base = B_row_offset * 2  # 0 or 2
        c_col_base = C_col_offset * 2  # 0, 2, or 4
        
        # A indices: i in {0,1}, j in {a_col_base, a_col_base+1}
        # A row index in U: i*m + j = i*4 + j
        # B indices: j in {b_row_base, b_row_base+1}, k in {c_col_base, c_col_base+1}
        # B row index in V: j*p + k = j*6 + k
        # C indices: k in {c_col_base, c_col_base+1}, i in {0,1}
        # C row index in W: k*n + i = k*2 + i
        
        # Strassen M1 = (A11+A22)*(B11+B22)
        # A11 = A[0, a_col_base], A22 = A[1, a_col_base+1]
        # B11 = B[b_row_base, c_col_base], B22 = B[b_row_base+1, c_col_base+1]
        c = start_col
        # U for M1: A11 and A22
        U[0*4 + a_col_base][c] = 1
        U[1*4 + a_col_base + 1][c] = 1
        # V for M1: B11 and B22
        V[b_row_base*6 + c_col_base][c] = 1
        V[(b_row_base+1)*6 + c_col_base + 1][c] = 1
        # W for M1: contributes to C11 and C22
        W[c_col_base*2 + 0][c] = 1   # C11 += M1
        W[(c_col_base+1)*2 + 1][c] = 1  # C22 += M1
        col += 1
        
        # M2 = (A21+A22)*B11
        # A21 = A[1, a_col_base], A22 = A[1, a_col_base+1]
        # B11 = B[b_row_base, c_col_base]
        c = start_col + 1
        U[1*4 + a_col_base][c] = 1
        U[1*4 + a_col_base + 1][c] = 1
        V[b_row_base*6 + c_col_base][c] = 1
        # W for M2: contributes to C21 and C22 (negative)
        W[(c_col_base)*2 + 1][c] = 1   # C21 += M2
        W[(c_col_base+1)*2 + 1][c] = -1  # C22 -= M2
        col += 1
        
        # M3 = A11*(B12-B22)
        # A11 = A[0, a_col_base]
        # B12 = B[b_row_base, c_col_base+1], B22 = B[b_row_base+1, c_col_base+1]
        c = start_col + 2
        U[0*4 + a_col_base][c] = 1
        V[b_row_base*6 + c_col_base + 1][c] = 1
        V[(b_row_base+1)*6 + c_col_base + 1][c] = -1
        # W for M3: contributes to C12 and C22
        W[(c_col_base+1)*2 + 0][c] = 1   # C12 += M3
        W[(c_col_base+1)*2 + 1][c] = 1   # C22 += M3
        col += 1
        
        # M4 = A22*(B21-B11)
        # A22 = A[1, a_col_base+1]
        # B21 = B[b_row_base+1, c_col_base], B11 = B[b_row_base, c_col_base]
        c = start_col + 3
        U[1*4 + a_col_base + 1][c] = 1
        V[(b_row_base+1)*6 + c_col_base][c] = 1
        V[b_row_base*6 + c_col_base][c] = -1
        # W for M4: contributes to C11 and C21
        W[c_col_base*2 + 0][c] = 1   # C11 += M4
        W[c_col_base*2 + 1][c] = 1   # C21 += M4
        col += 1
        
        # M5 = (A11+A12)*B22
        # A11 = A[0, a_col_base], A12 = A[0, a_col_base+1]
        # B22 = B[b_row_base+1, c_col_base+1]
        c = start_col + 4
        U[0*4 + a_col_base][c] = 1
        U[0*4 + a_col_base + 1][c] = 1
        V[(b_row_base+1)*6 + c_col_base + 1][c] = 1
        # W for M5: contributes to C11 (negative) and C12
        W[c_col_base*2 + 0][c] = -1  # C11 -= M5
        W[(c_col_base+1)*2 + 0][c] = 1  # C12 += M5
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        # A21 = A[1, a_col_base], A11 = A[0, a_col_base]
        # B11 = B[b_row_base, c_col_base], B12 = B[b_row_base, c_col_base+1]
        c = start_col + 5
        U[1*4 + a_col_base][c] = 1
        U[0*4 + a_col_base][c] = -1
        V[b_row_base*6 + c_col_base][c] = 1
        V[b_row_base*6 + c_col_base + 1][c] = 1
        # W for M6: contributes to C22
        W[(c_col_base+1)*2 + 1][c] = 1  # C22 += M6
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        # A12 = A[0, a_col_base+1], A22 = A[1, a_col_base+1]
        # B21 = B[b_row_base+1, c_col_base], B22 = B[b_row_base+1, c_col_base+1]
        c = start_col + 6
        U[0*4 + a_col_base + 1][c] = 1
        U[1*4 + a_col_base + 1][c] = -1
        V[(b_row_base+1)*6 + c_col_base][c] = 1
        V[(b_row_base+1)*6 + c_col_base + 1][c] = 1
        # W for M7: contributes to C11
        W[c_col_base*2 + 0][c] = 1  # C11 += M7
        col += 1
        
        return start_col + 7

    # Now add all 6 Strassen blocks
    # A_left (cols 0-1) * B_top (rows 0-1), 3 col blocks
    col = add_strassen_block(0, 0, 0, 0)
    col = add_strassen_block(0, 0, 1, col)
    col = add_strassen_block(0, 0, 2, col)
    # A_right (cols 2-3) * B_bottom (rows 2-3), 3 col blocks
    col = add_strassen_block(1, 1, 0, col)
    col = add_strassen_block(1, 1, 1, col)
    col = add_strassen_block(1, 1, 2, col)

    assert col == 42, f"Expected 42 columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

