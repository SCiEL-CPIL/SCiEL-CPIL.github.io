"""Rank-44 exact tensor decomposition for 3x4 @ 4x4 matrix multiplication."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    U = [[0.0] * 44 for _ in range(n * m)]
    V = [[0.0] * 44 for _ in range(m * p)]
    W = [[0.0] * 44 for _ in range(p * n)]
    col = 0

    # Top row of A (1x4) times B (4x4): schoolbook, 16 multiplications
    # U[i*m + j], V[j*p + k], W[k*n + i]
    for j in range(4):
        for k in range(4):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1
    # col = 16

    # Bottom 2x4 block of A times B (4x4)
    # A_bottom = [A21 (2x2) | A22 (2x2)]
    # B = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
    # C_bottom = [C21 (2x2), C22 (2x2)]
    # C21 = A21*B11 + A22*B21, C22 = A21*B12 + A22*B22
    # Use Strassen on the 2x2x2 blocks: rank 7 each, total 14

    # Strassen for C21 = A21*B11 + A22*B21
    # M1 = A21 * B11, M2 = A22 * B21, then C21 = M1 + M2
    # Actually, let's think about this differently.
    # C21 = A21*B11 + A22*B21 is a 2x2 matrix that's the sum of two 2x2 products.
    # We can compute each product separately: 4 multiplications each = 8 total.
    # Or use a rank-7 approach for the combined operation.
    
    # Let me use direct multiplication for the 2x4x4 bottom block:
    # 2*4*4 = 32? No, that's the full schoolbook for that sub-block.
    # But we already used 16 for the top. 16+32 = 48 (schoolbook total).
    
    # Better: use Strassen where possible.
    # For C21 = A21*B11 + A22*B21: treat as computing two 2x2 products.
    # Each 2x2 product has rank 7 (Strassen).
    # So C21 needs 7+7=14 multiplications.
    # C22 = A21*B12 + A22*B22: similarly 14 multiplications.
    # Total: 16 + 14 + 14 = 44.

    # Strassen for A21*B11 (rows 1,2 of A; cols 0,1 of B; output rows 1,2; cols 0,1)
    def strassen_block(a_rows, b_cols, out_rows, out_cols):
        nonlocal col
        # a_rows: 2 row indices in A (within the 2x4 bottom), b_cols: 2 col indices in B
        # out_rows: 2 row indices in output, out_cols: 2 col indices in output
        # A21 element at (dr, dc) where dr in {0,1}, dc in {0,1} corresponds to A row (1+dr), col dc
        # B11 element at (dc, dr) corresponds to B row dc, col (0+dr)
        # M1 = (A21 + A22)(B11 - B21) -- but we want separate products for the sum
        
        # Actually for C21 = A21*B11 + A22*B21, we can't directly use Strassen on the sum.
        # Let me just do direct multiplication for each 2x2 product.
        pass

    # Direct: A21 * B11
    for dr in range(2):
        for dc in range(2):
            for dk in range(2):
                # A row (1+dr), col dc; B row dk, col (0+dr); output row (1+dr), col dk... 
                # Wait, I need to be more careful.
                # C21[row, col] = sum over l of A_bottom[row, l] * B[l, col]
                # A_bottom row dr corresponds to A row (1+dr)
                # For C21, col dk corresponds to B col dk
                # A_bottom[row=dr, l] where l is 0..3
                # If l < 2: A[1+dr][l] = A21[dr][l]
                # If l >= 2: A[1+dr][l] = A22[dr][l-2]
                # B[l, col] where col is 0..3
                # For C21 (output cols 0,1): B[l][col]
                pass

    # Let me redo this more carefully with explicit indices.
    # I'll use direct schoolbook for the bottom 2x4x4 block but try to be smart.
    # Actually, let me just verify the rank-44 construction properly.

    # Reset
    col = 16  # after top row's 16 multiplications

    # For the bottom 2 rows of A (rows 1,2) times B (4x4):
    # Output rows 1,2 (in the 3x4 result)
    # For each output position (i, k) where i in {1,2}, k in {0,1,2,3}:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # Use Strassen-style for the 2x2 sub-blocks:
    # C21 (output rows 1,2; output cols 0,1) = A21*B11 + A22*B21
    # C22 (output rows 1,2; output cols 2,3) = A21*B12 + A22*B22
    
    # For C21 = A21*B11 + A22*B21:
    # Compute M1 = A21 * B11 using Strassen (7 mults)
    # Compute M2 = A22 * B21 using Strassen (7 mults)
    # C21 = M1 + M2 (additions, no new multiplications)
    
    def strassen_mult(a_row_offset, a_col_offset, b_row_offset, b_col_offset, out_row_offset, out_col_offset):
        """Compute 2x2 product: A[a_row_offset+a_row_offset+dr, a_col_offset+dc] * B[b_row_offset+dk, b_col_offset+dd]
        contributing to output at out_row_offset+dr, out_col_offset+dd.
        Uses Strassen's 7 multiplications."""
        nonlocal col
        # Strassen for C = A*B where A, B, C are 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6

        # For M1: (A11+A22)(B11+B22)
        # A11 is at (a_row_offset+0, a_col_offset+0), A22 at (a_row_offset+1, a_col_offset+1)
        # B11 at (b_row_offset+0, b_col_offset+0), B22 at (b_row_offset+1, b_col_offset+1)
        # This contributes to C11, C12, C21, C22 with coefficients
        # C11 += M1, C12 += -M1? No:
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        
        # Each M_i is a product of a linear combination of A entries and a linear combination of B entries.
        # For M1: A_part = A11 + A22, B_part = B11 + B22
        # In our tensor: U gets +1 for A11's index, +1 for A22's index
        # V gets +1 for B11's index, +1 for B22's index
        # W gets the coefficient for each output position

        # M1 contribution: C11 += 1*M1, C12 += 0, C21 += 0, C22 += 1*M1
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] += 1.0
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] += 1.0
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] += 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] += 1.0
        W[(out_col_offset + 0) * n + (out_row_offset + 0)][col] += 1.0
        W[(out_col_offset + 1) * n + (out_row_offset + 1)][col] += 1.0
        col += 1

        # M2: (A21+A22)B11
        # A21 at (a_row_offset+1, a_col_offset+0), A22 at (a_row_offset+1, a_col_offset+1)
        # B11 at (b_row_offset+0, b_col_offset+0)
        # C21 += M2
        U[(a_row_offset + 1) * m + (a_col_offset + 0)][col] += 1.0
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] += 1.0
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] += 1.0
        W[(out_col_offset + 0) * n + (out_row_offset + 1)][col] += 1.0
        col += 1

        # M3: A11(B21-B22)
        # A11 at (a_row_offset+0, a_col_offset+0)
        # B21 at (b_row_offset+1, b_col_offset+0), B22 at (b_row_offset+1, b_col_offset+1)
        # C12 += M3, C22 += M3
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] += 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 0)][col] += 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] -= 1.0
        W[(out_col_offset + 1) * n + (out_row_offset + 0)][col] += 1.0
        W[(out_col_offset + 1) * n + (out_row_offset + 1)][col] += 1.0
        col += 1

        # M4: A22(B11-B21)
        # A22 at (a_row_offset+1, a_col_offset+1)
        # B11 at (b_row_offset+0, b_col_offset+0), B21 at (b_row_offset+1, b_col_offset+0)
        # C11 += M4, C21 += M4
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] += 1.0
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] += 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 0)][col] -= 1.0
        W[(out_col_offset + 0) * n + (out_row_offset + 0)][col] += 1.0
        W[(out_col_offset + 0) * n + (out_row_offset + 1)][col] += 1.0
        col += 1

        # M5: (A11+A12)B22
        # A11 at (a_row_offset+0, a_col_offset+0), A12 at (a_row_offset+0, a_col_offset+1)
        # B22 at (b_row_offset+1, b_col_offset+1)
        # C11 -= M5, C12 += M5
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] += 1.0
        U[(a_row_offset + 0) * m + (a_col_offset + 1)][col] += 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] += 1.0
        W[(out_col_offset + 0) * n + (out_row_offset + 0)][col] -= 1.0
        W[(out_col_offset + 1) * n + (out_row_offset + 0)][col] += 1.0
        col += 1

        # M6: (A21-A11)(B11+B12)
        # A21 at (a_row_offset+1, a_col_offset+0), A11 at (a_row_offset+0, a_col_offset+0)
        # B11 at (b_row_offset+0, b_col_offset+0), B12 at (b_row_offset+0, b_col_offset+1)
        # C22 += M6
        U[(a_row_offset + 1) * m + (a_col_offset + 0)][col] += 1.0
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] -= 1.0
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] += 1.0
        V[(b_row_offset + 0) * p + (b_col_offset + 1)][col] += 1.0
        W[(out_col_offset + 1) * n + (out_row_offset + 1)][col] += 1.0
        col += 1

        # M7: (A12-A22)(B21+B22)
        # A12 at (a_row_offset+0, a_col_offset+1), A22 at (a_row_offset+1, a_col_offset+1)
        # B21 at (b_row_offset+1, b_col_offset+0), B22 at (b_row_offset+1, b_col_offset + 1)
        # C11 += M7
        U[(a_row_offset + 0) * m + (a_col_offset + 1)][col] += 1.0
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] -= 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 0)][col] += 1.0
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] += 1.0
        W[(out_col_offset + 0) * n + (out_row_offset + 0)][col] += 1.0
        col += 1

    # C21 = A21*B11 + A22*B21
    # A21: A rows 1,2 (0-indexed), cols 0,1
    # B11: B rows 0,1, cols 0,1
    # Output: rows 1,2, cols 0,1
    strassen_mult(1, 0, 0, 0, 1, 0)  # A21 * B11 -> contributes to C21
    strassen_mult(1, 2, 2, 0, 1, 0)  # A22 * B21 -> contributes to C21

    # C22 = A21*B12 + A22*B22
    # A21: A rows 1,2, cols 0,1
    # B12: B rows 0,1, cols 2,3
    # A22: A rows 1,2, cols 2,3
    # B22: B rows 2,3, cols 2,3
    # Output: rows 1,2, cols 2,3
    strassen_mult(1, 0, 0, 2, 1, 2)  # A21 * B12 -> contributes to C22
    strassen_mult(1, 2, 2, 2, 1, 2)  # A22 * B22 -> contributes to C22

    assert col == 44, f"Expected 44 columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
