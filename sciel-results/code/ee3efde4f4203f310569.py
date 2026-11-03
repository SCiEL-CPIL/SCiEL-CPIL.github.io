"""Rank-54 decomposition for 2x5 x 5x6 matrix multiplication using Strassen blocks."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]   # 10 x 54
    V = [[0.0] * rank for _ in range(m * p)]   # 30 x 54
    W = [[0.0] * rank for _ in range(p * n)]   # 12 x 54
    
    col = 0
    
    # Strassen 2x2x2: A(2x2)*B(2x2) = C(2x2)
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_block(a_row_off, b_row_off, c_row_off, c_col_off):
        """Add Strassen 2x2x2 block.
        a_row_off: row offset in U for A (2x2 submatrix)
        b_row_off: row offset in V for B (2x2 submatrix)  
        c_row_off: row offset in W for C (2x2 submatrix)
        c_col_off: column offset for the 2 columns of C
        Returns next column index.
        """
        nonlocal col
        # A indices: a_row_off, a_row_off+1 (rows), cols 0,1
        # B indices: b_row_off, b_row_off+1 (rows), cols 0,1
        # C indices: c_row_off, c_row_off+1 (rows), cols c_col_off, c_col_off+1
        
        # For U: row = a_row_off + r, where r in {0,1}, and we need to represent A[r][c]
        # U index: (a_row_off + r) * ? + c ... wait, U has shape (n*m, rank) = (10, rank)
        # U[row][col] where row indexes the A matrix elements
        # A is 2x2, so A[r][c] -> row = a_row_off + r*2 + c? No.
        # Actually U maps from A matrix to the tensor. Let me think about the convention.
        # U is (n*m, rank) = (10, 54). U[i*m+j][k] gives coefficient for A[i][j] in term k.
        # V is (m*p, rank) = (30, 54). V[j*p+k][c] gives coefficient for B[j][k] in term c.
        # W is (p*n, rank) = (12, 54). W[k*n+i][c] gives coefficient for C[i][k] in term c.
        # So for a 2x2x2 block at positions:
        # A rows: a_row_off, a_row_off+1 (these are the i indices, 0 and 1)
        # A cols: a_col_off, a_col_off+1 (these are the j indices)
        # So A[i][j] -> U[(a_row_off+i)*m + (a_col_off+j)][col]
        # B rows: b_row_off, b_row_off+1 (j indices)
        # B cols: b_col_off, b_col_off+1 (k indices)
        # B[j][k] -> V[(b_row_off+j)*p + (b_col_off+k)][col]
        # C rows: c_row_off, c_row_off+1 (k indices)
        # C cols: c_col_off, c_col_off+1 (i indices)
        # C[i][k] -> W[(c_row_off+k)*n + (c_col_off+i)][col]
        
        # Strassen variables for 2x2:
        # A00=A[0][0], A01=A[0][1], A10=A[1][0], A11=A[1][1]
        # B00=B[0][0], B01=B[0][1], B10=B[1][0], B11=B[1][1]
        
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B11+B01)
        
        def a_idx(r, c):
            return (a_row_off + r) * m + (a_col_off + c)
        
        def b_idx(r, c):
            return (b_row_off + r) * p + (b_col_off + c)
        
        def c_idx(k, i):
            return (c_row_off + k) * n + (c_col_off + i)
        
        # M1
        U[a_idx(0,0)][col] = 1
        U[a_idx(1,1)][col] = 1
        V[b_idx(0,0)][col] = 1
        V[b_idx(1,1)][col] = 1
        W[c_idx(0,0)][col] = 1
        W[c_idx(0,1)][col] = 0  # C01 doesn't use M1
        W[c_idx(1,0)][col] = 0
        W[c_idx(1,1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[a_idx(1,0)][col] = 1
        U[a_idx(1,1)][col] = 1
        V[b_idx(0,0)][col] = 1
        W[c_idx(1,0)][col] = 1
        W[c_idx(1,1)][col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[a_idx(0,0)][col] = 1
        V[b_idx(0,1)][col] = 1
        V[b_idx(1,1)][col] = -1
        W[c_idx(0,1)][col] = 1
        W[c_idx(1,1)][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[a_idx(1,1)][col] = 1
        V[b_idx(1,0)][col] = 1
        V[b_idx(0,0)][col] = -1
        W[c_idx(0,0)][col] = 1
        W[c_idx(1,0)][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[a_idx(0,0)][col] = 1
        U[a_idx(0,1)][col] = 1
        V[b_idx(1,1)][col] = 1
        W[c_idx(0,0)][col] = -1
        W[c_idx(0,1)][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B10)
        U[a_idx(1,0)][col] = 1
        U[a_idx(0,0)][col] = -1
        V[b_idx(0,0)][col] = 1
        V[b_idx(1,0)][col] = 1
        W[c_idx(1,1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B11+B01)
        U[a_idx(0,1)][col] = 1
        U[a_idx(1,1)][col] = -1
        V[b_idx(1,1)][col] = 1
        V[b_idx(0,1)][col] = 1
        W[c_idx(0,0)][col] = 1
        col += 1
        
        return col
    
    # Schoolbook 2x1x6 block
    def schoolbook_216(a_row_off, b_row_off, c_row_off, c_col_off):
        """A(2x1) * B(1x6) = C(2x6). 12 terms.
        a_row_off: row offset in A (i index)
        b_row_off: row offset in B (j index, only j=0)
        c_row_off: row offset in C (k index)
        c_col_off: column offset for 6 columns of C
        """
        nonlocal col
        for i in range(2):
            for k in range(6):
                U[(a_row_off + i) * m + (0)][col] = 1  # A[i][0]
                V[(b_row_off + 0) * p + k][col] = 1     # B[0][k]
                W[(c_row_off + k) * n + c_col_off + i][col] = 1  # C[i][k]
                col += 1
        return col
    
    # Now build the full decomposition:
    # A is 2x5, B is 5x6, C is 2x6
    # Split A = [A1 | A2] where A1 is 2x1, A2 is 2x4
    # Split B = [B1; B2] where B1 is 1x6, B2 is 4x6
    # C = A1*B1 + A2*B2
    
    # Part 1: A1(2x1) * B1(1x6) -> C1(2x6), rank 12
    # A1 uses A rows 0,1 and col 0
    # B1 uses B row 0 and cols 0-5
    # C1 uses all of C
    col = schoolbook_216(0, 0, 0, 0)
    # col should be 12 now
    
    # Part 2: A2(2x4) * B2(4x6) -> C2(2x6), rank 42
    # Split A2 into A2a(2x2) and A2b(2x2)
    # A2a: A rows 0,1, cols 1,2
    # A2b: A rows 0,1, cols 3,4
    # Split B2 into B2a(2x6) and B2b(2x6)
    # B2a: B rows 1,2, cols 0-5
    # B2b: B rows 3,4, cols 0-5
    # C2 = A2a*B2a + A2b*B2b
    # Each 2x2x6 uses 3 Strassen 2x2x2 blocks
    
    # A2a * B2a: A2a rows 0,1 cols 1,2; B2a rows 1,2 cols 0-5; C all
    # Split into 3 Strassen blocks for columns 0-1, 2-3, 4-5
    col = strassen_block(0, 1, 0, 0)   # A2a, B2a, C cols 0-1
    col = strassen_block(0, 1, 2, 2)   # A2a, B2a, C cols 2-3  
    col = strassen_block(0, 1, 4, 4)   # A2a, B2a, C cols 4-5
    
    # A2b * B2b: A2b rows 0,1 cols 3,4; B2b rows 3,4 cols 0-5; C all
    col = strassen_block(0, 3, 0, 0)   # A2b, B2b, C cols 0-1
    col = strassen_block(0, 3, 2, 2)   # A2b, B2b, C cols 2-3
    col = strassen_block(0, 3, 4, 4)   # A2b, B2b, C cols 4-5
    
    assert col == 54, f"Expected rank 54, got {col}"
    
    # Convert to proper integer-like representation
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
