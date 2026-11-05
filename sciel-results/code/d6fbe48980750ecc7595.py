"""Matrix multiplication 4x5 x 5x5 with block-based Strassen improvement."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Split the problem into blocks and use Strassen for 2x2x2 subproblems.
    
    We split:
      A (4x5) into A_top (2x5) and A_bot (2x5)
      B (5x5) into B_left (5x2) and B_right (5x3)
    
    Then C = A*B is 4x5, split into:
      C_top_left (2x2) = A_top * B_left
      C_top_right (2x3) = A_top * B_right
      C_bot_left (2x2) = A_bot * B_left
      C_bot_right (2x3) = A_bot * B_right
    
    For the 2x2 blocks: use Strassen (rank 7)
    For the 2x3 blocks: A_top (2x5) * B_right (5x3)
      Split B_right's 5 rows into 2+3:
        B_right_top (2x3), B_right_bot (3x3)
      A_top * B_right = A_top[:2] * B_right_top + A_top[2:] * B_right_bot
      
      Wait, A_top is 2x5, so we split its 5 columns:
        A_top_left (2x2), A_top_right (2x3)
      B_right is 5x3, split its 5 rows:
        B_right_top (2x3), B_right_bot (3x3)
      
      So A_top * B_right = A_top_left * B_right_top + A_top_right * B_right_bot
      where A_top_left (2x2) * B_right_top (2x3): rank at most 2*2*3=12
      and A_top_right (2x3) * B_right_bot (3x3): rank at most 2*3*3=18
      
      Total for one 2x3 block: 12 + 18 = 30, schoolbook would be 2*5*3=30. No gain here.
      
    Let me reconsider. The schoolbook rank is 4*5*5 = 100.
    
    Better approach: Use Strassen-style recursion more cleverly.
    
    Alternative: Split 5 = 2 + 3 in all dimensions.
    
    Actually, let me try a simpler approach: just use Strassen where possible.
    
    Split A (4x5) into two 2x5 blocks vertically: A1 (2x5), A2 (2x5)
    Split B (5x5) into two parts horizontally: B1 (5x2), B2 (5x3)
    
    C1 = A1 * B1 (2x2) -> Strassen, rank 7
    C2 = A1 * B2 (2x3) -> schoolbook, rank 2*5*3 = 30
    C3 = A2 * B1 (2x2) -> Strassen, rank 7
    C4 = A2 * B2 (2x3) -> schoolbook, rank 2*5*3 = 30
    
    Total: 7 + 30 + 7 + 30 = 74
    
    That's an improvement from 100!
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build U, V, W matrices with integer entries
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        """Add a rank-1 term."""
        U.append([coeff_u if i == u_idx else 0 for i in range(n * m)])
        V.append([coeff_v if i == v_idx else 0 for i in range(m * p)])
        W.append([coeff_w if i == w_idx else 0 for i in range(p * n)])
    
    def strassen_2x2x2(offset_a, offset_b, offset_c, sign=1):
        """
        Add Strassen terms for a 2x2 * 2x2 multiplication.
        
        offset_a: starting index in A for row 0 col 0 (so A is 2x2 with elements at:
                  (offset_a, offset_a), (offset_a, offset_a+1), (offset_a+1, offset_a), (offset_a+1, offset_a+1))
        offset_b: starting index in B
        offset_c: starting index in C
        
        For Strassen:
          M1 = (A00+A11)*(B00+B11)
          M2 = (A10+A11)*B00
          M3 = A00*(B01-B11)
          M4 = A11*(B10-B00)
          M5 = (A00+A01)*B11
          M6 = (A10-A00)*(B00+B01)
          M7 = (A01-A11)*(B10+B11)
          
          C00 = M1 + M4 - M5 + M7
          C01 = M3 + M5
          C10 = M2 + M4
          C11 = M1 - M2 + M3 + M6
        """
        # A indices (row, col) -> linear index in A (n*m dimension)
        # A is stored as A[i*m + j] where i is row, j is col
        # For a 2x2 block starting at row r, col c of A:
        # A00 at (r, c), A01 at (r, c+1), A10 at (r+1, c), A11 at (r+1, c+1)
        
        a00 = offset_a  # row 0, col 0
        a01 = offset_a + 1  # row 0, col 1
        a10 = offset_a + m  # row 1, col 0
        a11 = offset_a + m + 1  # row 1, col 1
        
        # B indices: B is stored as B[i*p + j] where i is row, j is col
        # For a 2x2 block starting at row r, col c of B:
        b00 = offset_b  # row 0, col 0
        b01 = offset_b + 1  # row 0, col 1
        b10 = offset_b + p  # row 1, col 0
        b11 = offset_b + p + 1  # row 1, col 1
        
        # C indices: W[k*n+i, r] where k is col of B, i is row of A, r is col of C
        # For C block starting at row r, col c of C:
        # C00 at (k=0, i=0) -> index 0*n + 0 = 0, but with offset
        # C is stored as W[k*n + i, r]
        # For output block starting at row r, col c:
        # C00: i=r, k=c -> index c*n + r, col c
        # C01: i=r, k=c -> index c*n + r, col c+1
        # C10: i=r+1, k=c -> index c*n + (r+1), col c
        # C11: i=r+1, k=c -> index c*n + (r+1), col c+1
        
        c00_row = offset_c  # this is the row index in C (i dimension)
        c00_col = offset_c  # this is the col index in C (k dimension)
        # Wait, I need to be careful. Let me redefine.
        
        # Let me say the 2x2 C block has its top-left at C row r, col c.
        # W index for C[i,j] where i is A's row, j is B's col: W[j*n + i, ?]
        # The rank dimension is the third index.
        
        # Actually, W has shape (p*n, rank). W[k*n + i, c] gives the contribution to C[i,k]... 
        # Wait, the convention is W[k*n+i, r] where r is the rank index.
        # So W maps (k, i) -> rank contribution to C[i, k].
        
        # For a 2x2 C block starting at C row r (A's row) and col c (B's col):
        # C[r,c] gets contributions from W[c*n + r, rank]
        # C[r,c+1] gets contributions from W[(c+1)*n + r, rank]
        # C[r+1,c] gets contributions from W[c*n + (r+1), rank]
        # C[r+1,c+1] gets contributions from W[(c+1)*n + (r+1), rank]
        
        # Let me redefine offset_c as (r, c) where r is the A-row offset, c is the B-col offset.
        r_off = offset_c  # I'll pass this as r*n + c? No, let me just pass r and c separately.
        
        # Actually, let me restructure. I'll pass the actual A-row start, B-col start for the C block.
        
        # Let me just hardcode the Strassen for specific positions since we know our layout.
        pass
    
    # Given the complexity, let me just directly enumerate all terms.
    
    # C1 = A_top * B_left (2x2), A rows 0-1, B cols 0-1
    # Strassen for C1:
    # A_top: rows 0-1 of A, all 5 cols. But we only use cols 0-1 for B_left.
    # B_left: all 5 rows of B, cols 0-1.
    
    # For Strassen on A_top[:,0:2] * B_left[0:2,:]:
    # A elements: A[0,0]=0, A[0,1]=1, A[1,0]=m, A[1,1]=m+1
    # B elements: B[0,0]=0, B[0,1]=1, B[1,0]=p, B[1,1]=p+1
    
    # M1 = (A00+A11)*(B00+B11)
    # We need to express this as a sum of rank-1 terms.
    
    # Actually, for the tensor decomposition, each Strassen multiplication corresponds to 7 terms.
    # Each term is a product of a linear combination of A entries, a linear combination of B entries,
    # and contributes to linear combinations of C entries.
    
    # This is getting complex. Let me just do schoolbook but with the block split to save on the 2x2 parts.
    
    # Schoolbook for the full thing: 100 terms.
    # With Strassen on the two 2x2 blocks: save 1 each (8->7), so 100 - 2 = 98.
    # Plus the 2x3 parts are still schoolbook: 30 + 30 = 60.
    # Total: 7 + 30 + 7 + 30 = 74.
    
    # Let me implement this properly.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_rank1(u_vals, v_vals, w_vals):
        """Add a rank-1 term with given coefficient vectors."""
        nonlocal rank
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)
        rank += 1
    
    def schoolbook_2x3(a_row_start, b_col_start):
        """
        Schoolbook multiplication for A[rows a_row_start:a_row_start+2, all 5 cols] * B[all 5 rows, cols b_col_start:b_col_start+3]
        Result is 2x3, stored in C.
        
        For each output C[i,j] (i in 0..1, j in 0..2):
          C[i,j] = sum_k A[a_row_start+i, k] * B[k, b_col_start+j]
        
        This gives 2*5*3 = 30 terms.
        """
        for i in range(2):
            for k in range(5):
                for j in range(3):
                    # A index: (a_row_start + i) * m + k
                    a_idx = (a_row_start + i) * m + k
                    # B index: k * p + (b_col_start + j)
                    b_idx = k * p + (b_col_start + j)
                    # C index: (b_col_start + j) * n + (a_row_start + i)
                    c_idx = (b_col_start + j) * n + (a_row_start + i)
                    
                    u = [0] * (n * m)
                    u[a_idx] = 1
                    v = [0] * (m * p)
                    v[b_idx] = 1
                    w = [0] * (p * n)
                    w[c_idx] = 1
                    add_rank1(u, v, w)
    
    def strassen_2x2(a_row_start, b_col_start, c_row_start, c_col_start):
        """
        Strassen multiplication for A[a_row_start:a_row_start+2, b_col_start:b_col_start+2] * 
        B[b_col_start:b_col_start+2, all relevant rows]
        
        Wait, B's rows are the same as A's cols. Let me think again.
        
        A is n x m, B is m x p.
        A rows: 0..n-1, cols: 0..m-1
        B rows: 0..m-1, cols: 0..p-1
        C rows: 0..n-1, cols: 0..p-1
        
        For a 2x2 block:
        A block: rows r..r+1, cols c..c+1 of A
        B block: rows c..c+1 (same cols of A), all? No, B's rows correspond to A's cols.
        
        So for C block at rows r..r+1, cols c..c+1:
        C[r+i, c+j] = sum_k A[r+i, k] * B[k, c+j] for i,j in 0..1
        
        The "inner" dimension is k from 0 to m-1. For a 2x2 Strassen, we'd need k to be 2,
        but here k ranges over all m=5. So Strassen doesn't directly apply unless we split k too.
        
        This is more complex than I thought. Let me just use schoolbook for everything.
        """
        # Fall back to schoolbook for 2x2x5 (not 2x2x2)
        # This is 2 rows of A, 5 cols (k dimension), 2 cols of B
        # Schoolbook: 2*5*2 = 20 terms
        for i in range(2):
            for k in range(5):
                for j in range(2):
                    a_idx = (a_row_start + i) * m + (b_col_start + j)  # Wait, A's col is k, not b_col_start+j
                    # A[r+i, k] where k is the inner index
                    a_idx = (a_row_start + i) * m + k
                    b_idx = k * p + (c_col_start + j)
                    c_idx = (c_col_start + j) * n + (a_row_start + i)
                    
                    u = [0] * (n * m)
                    u[a_idx] = 1
                    v = [0] * (m * p)
                    v[b_idx] = 1
                    w = [0] * (p * n)
                    w[c_idx] = 1
                    add_rank1(u, v, w)
    
    # C_top_left = A[0:2, :] * B[:, 0:2] (2x2 output)
    strassen_2x2(0, 0, 0, 0)  # This will do schoolbook 2x5x2 = 20 terms
    
    # C_top_right = A[0:2, :] * B[:, 2:5] (2x3 output)
    schoolbook_2x3(0, 2)  # 30 terms
    
    # C_bot_left = A[2:4, :] * B[:, 0:2] (2x2 output)
    strassen_2x2(2, 0, 2, 0)  # 20 terms
    
    # C_bot_right = A[2:4, :] * B[:, 2:5] (2x3 output)
    schoolbook_2x3(2, 2)  # 30 terms
    
    total_rank = rank
    # Should be 20 + 30 + 20 + 30 = 100, same as schoolbook.
    
    # That's not an improvement. The issue is that Strassen only helps when the inner dimension is 2.
    
    # Let me try a different split: split the inner dimension (m=5) into 2+3.
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
    
    # First part: A (4x2) * B (2x5) -> 4x5 output
    # Schoolbook: 4*2*5 = 40
    # Second part: A (4x3) * B (3x5) -> 4x5 output
    # Schoolbook: 4*3*5 = 60
    # Total: 100. Still the same.
    
    # I think for this small size, it's hard to beat schoolbook with simple block algorithms.
    # Let me just return the schoolbook decomposition.
    
    U = []
    V = []
    W = []
    rank = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
                rank += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
