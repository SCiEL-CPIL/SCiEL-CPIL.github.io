"""Matrix multiplication 4,4,9 - rank 108 decomposition using Strassen for 2x2x2 sub-blocks."""
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-108 exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    - Split the 4x4 matrix A into four 2x2 blocks: A00, A01, A10, A11
    - Split the 4x9 matrix B into two 2x9 blocks: B0, B1
    - Split each 2x9 block into three 2x3 sub-blocks: B{i,0}, B{i,1}, B{i,2}
    - Split each 2x3 sub-block into one 2x2 and one 2x1 part
    - Use Strassen (rank 7) for each 2x2x2 multiplication
    - Use schoolbook (rank 2) for each 2x2x1 multiplication
    - Total rank: 4 * (3 * (7 + 2)) = 4 * 27 = 108
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Build up the decomposition step by step
    # We'll construct U, V, W as lists of lists
    
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    rank = 108
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # Strassen 2x2x2: 7 multiplications
    # For A (2x2) * B (2x2) = C (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Strassen coefficients for U, V, W
    # U rows for a 2x2 A block: rows (i0*2+j0, i0*2+j1, i1*2+j0, i1*2+j1)
    # V rows for a 2x2 B block: rows (j0*2+k0, j0*2+k1, j1*2+k0, j1*2+k1)
    # W rows for a 2x2 C block: rows (k0*2+i0, k0*2+i1, k1*2+i0, k1*2+i1)
    
    def add_strassen_222(u_base, v_base, w_base, offset):
        """Add Strassen 2x2x2 decomposition starting at column offset."""
        # M1: (A00+A11)(B00+B11)
        # U: A00 and A11 get +1
        U[u_base + 0][offset] = 1   # A00
        U[u_base + 3][offset] = 1   # A11
        V[v_base + 0][offset] = 1   # B00
        V[v_base + 3][offset] = 1   # B11
        # W: C00 gets +1, C11 gets +1
        W[w_base + 0][offset] = 1   # C00 (k=0,i=0)
        W[w_base + 3][offset] = 1   # C11 (k=1,i=1)
        
        # M2: (A10+A11)B00
        U[u_base + 2][offset+1] = 1   # A10
        U[u_base + 3][offset+1] = 1   # A11
        V[v_base + 0][offset+1] = 1   # B00
        # W: C10 gets +1
        W[w_base + 2][offset+1] = 1   # C10 (k=1,i=0)
        
        # M3: A00(B01-B11)
        U[u_base + 0][offset+2] = 1   # A00
        V[v_base + 1][offset+2] = 1   # B01
        V[v_base + 3][offset+2] = -1  # B11
        # W: C01 gets +1, C11 gets +1
        W[w_base + 1][offset+2] = 1   # C01 (k=0,i=1)
        W[w_base + 3][offset+2] = 1   # C11 (k=1,i=1)
        
        # M4: A11(B10-B00)
        U[u_base + 3][offset+3] = 1   # A11
        V[v_base + 2][offset+3] = 1   # B10
        V[v_base + 0][offset+3] = -1  # B00
        # W: C00 gets +1, C10 gets +1
        W[w_base + 0][offset+3] = 1   # C00 (k=0,i=0)
        W[w_base + 2][offset+3] = 1   # C10 (k=1,i=0)
        
        # M5: (A00+A01)B11
        U[u_base + 0][offset+4] = 1   # A00
        U[u_base + 1][offset+4] = 1   # A01
        V[v_base + 3][offset+4] = 1   # B11
        # W: C00 gets -1, C01 gets +1
        W[w_base + 0][offset+4] = -1  # C00 (k=0,i=0)
        W[w_base + 1][offset+4] = 1   # C01 (k=0,i=1)
        
        # M6: (A10-A00)(B00+B01)
        U[u_base + 2][offset+5] = 1   # A10
        U[u_base + 0][offset+5] = -1  # A00
        V[v_base + 0][offset+5] = 1   # B00
        V[v_base + 1][offset+5] = 1   # B01
        # W: C11 gets +1
        W[w_base + 3][offset+5] = 1   # C11 (k=1,i=1)
        
        # M7: (A01-A11)(B10+B11)
        U[u_base + 1][offset+6] = 1   # A01
        U[u_base + 3][offset+6] = -1  # A11
        V[v_base + 2][offset+6] = 1   # B10
        V[v_base + 3][offset+6] = 1   # B11
        # W: C00 gets +1
        W[w_base + 0][offset+6] = 1   # C00 (k=0,i=0)
        
        return offset + 7
    
    def add_schoolbook_221(u_base, v_base, w_base, offset):
        """Add schoolbook 2x2x1 decomposition (2 multiplications) starting at column offset."""
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # U rows: A00, A01, A10, A11
        # V rows: B0, B1
        # W rows: C0, C1
        
        # M1: A00*B0
        U[u_base + 0][offset] = 1   # A00
        V[v_base + 0][offset] = 1   # B0
        W[w_base + 0][offset] = 1   # C0
        
        # M2: A01*B1
        U[u_base + 1][offset+1] = 1   # A01
        V[v_base + 1][offset+1] = 1   # B1
        W[w_base + 0][offset+1] = 1   # C0
        
        # M3: A10*B0
        U[u_base + 2][offset+2] = 1   # A10
        V[v_base + 0][offset+2] = 1   # B0
        W[w_base + 1][offset+2] = 1   # C1
        
        # M4: A11*B1
        U[u_base + 3][offset+3] = 1   # A11
        V[v_base + 1][offset+3] = 1   # B1
        W[w_base + 1][offset+3] = 1   # C1
        
        return offset + 4
    
    # Now combine all blocks
    # A is 4x4, split into 2x2 blocks:
    # A00: rows 0-1, cols 0-1 -> U rows 0-3
    # A01: rows 0-1, cols 2-3 -> U rows 4-7
    # A10: rows 2-3, cols 0-1 -> U rows 8-11
    # A11: rows 2-3, cols 2-3 -> U rows 12-15
    
    # B is 4x9, split into:
    # B0: rows 0-1, all 9 cols -> V rows 0-17 (9 cols)
    # B1: rows 2-3, all 9 cols -> V rows 18-35 (9 cols)
    
    # Each 2x9 B block split into three 2x3 blocks:
    # B{i,0}: cols 0-2 -> V cols 0-5 (within the 2x9 block)
    # B{i,1}: cols 3-5 -> V cols 6-11
    # B{i,2}: cols 6-8 -> V cols 12-17
    
    # Each 2x3 split into 2x2 (cols 0-1) and 2x1 (col 2)
    
    # C is 9x4, with W[k*n+i] = W[k*4+i]
    # Output C is 4x9: C[i,k] for i in 0..3, k in 0..8
    # But W uses trace convention: W[k*n+i, r] where n=4
    # So W row index = k*4 + i
    
    # For A00 * B0: output is 2x9, goes to C[0:2, 0:9]
    # For A01 * B1: output is 2x9, goes to C[0:2, 2:5]? No...
    
    # Wait, let me reconsider the block structure.
    # A (4x4) * B (4x9) = C (4x9)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split A into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B0], [B1]] where B0 is 2x9, B1 is 2x9
    
    # C = [[A00*B0 + A01*B1], [A10*B0 + A11*B1]]
    # C[0:2, :] = A00*B0 + A01*B1  (top 2 rows)
    # C[2:4, :] = A10*B0 + A11*B1  (bottom 2 rows)
    
    # So we need to compute:
    # C_top = A00*B0 + A01*B1
    # C_bot = A10*B0 + A11*B1
    
    # Each is a sum of two 2x9 products, each using 2x2 blocks.
    
    # For C_top:
    # A00*B0: A00 is 2x2, B0 is 2x9
    # Split B0 into three 2x3 blocks: B0[0:2,0:3], B0[0:2,3:6], B0[0:2,6:9]
    # Each 2x3 = 2x2 + 2x1
    
    # Let's do it systematically:
    
    def add_2x9_product(a_block, b_block, c_row_offset, c_col_offset, col_start):
        """
        Add decomposition for (2x2 A block) * (2x9 B block) -> (2x9 C block)
        a_block: (row_start, col_start) of A block in original A
        b_block: (row_start, col_start) of B block in original B
        c_row_offset: row offset in C
        c_col_offset: col offset in C
        col_start: starting column in decomposition
        """
        a_r, a_c = a_block
        b_r, b_c = b_block
        
        # U rows for this A block: a_r*4 + a_c + [0,1,2,3]
        u_base = a_r * m + a_c  # This gives the row in U for the top-left of the 2x2 block
        
        # V rows for this B block: b_r*p + b_c + [0..8]
        v_base = b_r * p + b_c
        
        # W rows: for output C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+9]
        # W[k*n+i] where k is col in C, i is row in C
        # So for row i in C (absolute), col k in C (absolute): W index = k*n + i
        
        col = col_start
        
        # Split B block into three 2x3 sub-blocks
        for s in range(3):
            # B sub-block: 2x3, cols b_c + 3*s to b_c + 3*s + 2
            # Split into 2x2 (cols 0-1) and 2x1 (col 2)
            
            # 2x2 part: cols b_c + 3*s to b_c + 3*s + 1
            v_2x2_base = b_r * p + b_c + 3 * s
            # Output cols in C: c_col_offset + 3*s to c_col_offset + 3*s + 1
            w_2x2_base = (c_col_offset + 3 * s) * n + c_row_offset
            
            col = add_strassen_222(u_base, v_2x2_base, w_2x2_base, col)
            
            # 2x1 part: col b_c + 3*s + 2
            v_2x1_base = b_r * p + b_c + 3 * s + 2
            # Output col in C: c_col_offset + 3*s + 2
            w_2x1_base = (c_col_offset + 3 * s + 2) * n + c_row_offset
            
            col = add_schoolbook_221(u_base, v_2x1_base, w_2x1_base, col)
        
        return col
    
    # Now compute all four 2x9 products:
    
    # C_top = A00*B0 + A01*B1
    # A00: rows 0-1, cols 0-1 -> a_block = (0,0)
    # B0: rows 0-1, cols 0-8 -> b_block = (0,0)
    # C top rows: 0-1, cols 0-8
    col = add_2x9_product((0,0), (0,0), 0, 0, 0)
    
    # A01: rows 0-1, cols 2-3 -> a_block = (0,2)
    # B1: rows 2-3, cols 0-8 -> b_block = (2,0)
    # C top rows: 0-1, cols 0-8
    col = add_2x9_product((0,2), (2,0), 0, 0, col)
    
    # C_bot = A10*B0 + A11*B1
    # A10: rows 2-3, cols 0-1 -> a_block = (2,0)
    # B0: rows 0-1, cols 0-8 -> b_block = (0,0)
    # C bottom rows: 2-3, cols 0-8
    col = add_2x9_product((2,0), (0,0), 2, 0, col)
    
    # A11: rows 2-3, cols 2-3 -> a_block = (2,2)
    # B1: rows 2-3, cols 0-8 -> b_block = (2,0)
    # C bottom rows: 2-3, cols 0-8
    col = add_2x9_product((2,2), (2,0), 2, 0, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

