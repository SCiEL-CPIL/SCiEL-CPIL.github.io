"""Program for 3x5 by 5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a block structure:
    # A is 3x5, B is 5x5.
    # We can split A into 3 blocks of 1x5.
    # We can split B into 5 blocks of 5x1.
    # This is just the trivial decomposition with rank 3*5*5 = 75.
    
    # To improve, we can use Strassen for 2x2 blocks.
    # Let's try to decompose the 3x5 * 5x5 multiplication.
    # We can view the 3x5 matrix A as [A1; A2; A3] where Ai is 1x5.
    # We can view the 5x5 matrix B as [B1 B2 B3 B4 B5] where Bj is 5x1.
    # C = A * B = [A1*B; A2*B; A3*B]. Each Ai*B is 1x5.
    # This doesn't help directly with Strassen.
    
    # Let's try a different blocking.
    # Split A into 2x2 blocks? No, 3 is not even.
    # We can split the 5x5 B into 2x2, 2x2, 2x2, 1x1, 1x1? No.
    # Split the 5 dimension into 2+3.
    # Let's split the inner dimension (5) into 2 and 3.
    # A = [A_2 | A_3] where A_2 is 3x2 and A_3 is 3x3.
    # B = [B_2; B_3] where B_2 is 2x5 and B_3 is 3x5.
    # C = A_2 * B_2 + A_3 * B_3.
    
    # A_2 * B_2 is 3x2 * 2x5 = 3x5.
    # A_3 * B_3 is 3x3 * 3x5 = 3x5.
    
    # Let's compute A_2 * B_2.
    # A_2 is 3x2, B_2 is 2x5.
    # We can use Strassen for 2x2 blocks if we pad?
    # Or just use trivial for 3x2 * 2x5? Rank = 3*2*5 = 30.
    # Or split 3 into 1+2?
    # A_2 = [a1; A_1x2] where a1 is 1x2, A_1x2 is 2x2.
    # A_1x2 * B_2 can use Strassen? B_2 is 2x5.
    # This is getting complicated.
    
    # Let's stick to a simpler improvement:
    # Use Strassen for the 2x2 sub-blocks where possible.
    # The dimensions are 3, 5, 5.
    # We can decompose the 5x5 multiplication into:
    # - A 2x2 block (top-left of B, top 2 rows of A) -> Strassen (rank 7)
    # - A 2x3 block (top 2 rows of A, right 3 cols of B) -> ?
    # - A 1x2 block (bottom row of A, left 2 cols of B) -> ?
    # - A 1x3 block (bottom row of A, right 3 cols of B) -> ?
    # - A 3x2 block (bottom 3 rows of A? No, A is 3x5. Rows 0,1,2. Cols 0,1,2,3,4.)
    # Let's index A as 3 rows, 5 cols. B as 5 rows, 5 cols.
    # Split A rows into 2 and 1. Split A cols into 2 and 3.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 1x2, A22 is 1x3.
    # Split B rows into 2 and 3. Split B cols into 2 and 3.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3.
    
    # C = A * B
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x3) = A11*B12 + A12*B22
    # C21 (1x2) = A21*B11 + A22*B21
    # C22 (1x3) = A21*B12 + A22*B22
    
    # We need to sum these up.
    # Let's compute each term.
    # Term 1: A11 (2x2) * B11 (2x2) -> 2x2. Use Strassen (rank 7).
    # Term 2: A12 (2x3) * B21 (3x2) -> 2x2. Use Strassen? No, 3 is not 2.
    #   We can pad 2x3 to 2x4? No.
    #   We can use trivial for 2x3 * 3x2. Rank = 2*3*2 = 12.
    #   Or split 3 into 1+2.
    #   A12 = [A12a (2x1) | A12b (2x2)]
    #   B21 = [B21a (1x2); B21b (2x2)]
    #   A12*B21 = A12a*B21a + A12b*B21b.
    #   A12a (2x1) * B21a (1x2) -> 2x2. Trivial rank 2*1*2 = 4.
    #   A12b (2x2) * B21b (2x2) -> 2x2. Strassen rank 7.
    #   Total for Term 2: 4 + 7 = 11.
    
    # Term 3: A11 (2x2) * B12 (2x3) -> 2x3.
    #   Split B12 into B12a (2x1) and B12b (2x2).
    #   A11*B12a (2x1) -> 2x1. Trivial rank 2*2*1 = 4.
    #   A11*B12b (2x2) -> 2x2. Strassen rank 7.
    #   Total for Term 3: 4 + 7 = 11.
    
    # Term 4: A21 (1x2) * B11 (2x2) -> 1x2.
    #   Trivial rank 1*2*2 = 4.
    
    # Term 5: A22 (1x3) * B21 (3x2) -> 1x2.
    #   Split B21 into B21a (1x2) and B21b (2x2).
    #   A22*B21a (1x2) -> 1x2. Trivial rank 1*3*2 = 6.
    #   Wait, A22 is 1x3. B21a is 1x2. Inner dim 3 vs 1? No.
    #   A22 is 1x3. B21 is 3x2.
    #   Split A22 into A22a (1x1) and A22b (1x2).
    #   Split B21 into B21a (1x2) and B21b (2x2).
    #   A22a (1x1) * B21a (1x2) -> 1x2. Rank 1*1*2 = 2.
    #   A22b (1x2) * B21b (2x2) -> 1x2. Rank 1*2*2 = 4.
    #   Total for Term 5: 2 + 4 = 6.
    
    # Term 6: A21 (1x2) * B12 (2x3) -> 1x3.
    #   Split B12 into B12a (2x1) and B12b (2x2).
    #   A21 (1x2) * B12a (2x1) -> 1x1. Rank 1*2*1 = 2.
    #   A21 (1x2) * B12b (2x2) -> 1x2. Rank 1*2*2 = 4.
    #   Total for Term 6: 2 + 4 = 6.
    
    # Term 7: A22 (1x3) * B22 (3x3) -> 1x3.
    #   Split A22 into A22a (1x1) and A22b (1x2).
    #   Split B22 into B22a (1x3) and B22b (2x3).
    #   A22a (1x1) * B22a (1x3) -> 1x3. Rank 1*1*3 = 3.
    #   A22b (1x2) * B22b (2x3) -> 1x3. Rank 1*2*3 = 6.
    #   Total for Term 7: 3 + 6 = 9.
    
    # Total Rank = 7 (T1) + 11 (T2) + 11 (T3) + 4 (T4) + 6 (T5) + 6 (T6) + 9 (T7)
    # Total = 7 + 11 + 11 + 4 + 6 + 6 + 9 = 54.
    # This is a significant improvement over 75.
    
    # Let's implement this.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_trivial(n, m, p, a_offset, b_offset, c_offset):
        """
        Adds trivial decomposition for n x m * m x p.
        a_offset: (row, col) offset in A
        b_offset: (row, col) offset in B
        c_offset: (row, col) offset in C
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    
                    # U index: (a_row + i, a_col + j)
                    u_idx = (a_offset[0] + i) * m + (a_offset[1] + j)
                    u[u_idx] = 1
                    
                    # V index: (b_row + j, b_col + k)
                    v_idx = (b_offset[0] + j) * p + (b_offset[1] + k)
                    v[v_idx] = 1
                    
                    # W index: (c_col + k, c_row + i) -> W[k*n+i]
                    # Wait, W shape is (p*n). Index is k*n + i.
                    # c_offset is (row, col) in C.
                    # C row is i, C col is k.
                    # So W index is (c_offset[1] + k) * n + (c_offset[0] + i)
                    w_idx = (c_offset[1] + k) * n + (c_offset[0] + i)
                    w[w_idx] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    def add_strassen(n, m, p, a_offset, b_offset, c_offset):
        """
        Adds Strassen decomposition for 2x2 * 2x2.
        Assumes n=2, m=2, p=2.
        """
        # Strassen formulas for 2x2 matrices A, B.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 columns.
        # Each column corresponds to one M.
        # We need to construct U, V, W for each M.
        
        # Helper to add a term to U, V, W
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: list of (idx, coeff)
            # v_coeffs: list of (idx, coeff)
            # w_coeffs: list of (idx, coeff)
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            
            for idx, coeff in u_coeffs:
                u[idx] += coeff
            for idx, coeff in v_coeffs:
                v[idx] += coeff
            for idx, coeff in w_coeffs:
                w[idx] += coeff
                
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
            
        # Indices in local 2x2 blocks
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Global indices
        def a_idx(i, j):
            return (a_offset[0] + i) * m + (a_offset[1] + j)
        def b_idx(i, j):
            return (b_offset[0] + i) * p + (b_offset[1] + j)
        def c_idx(i, j):
            return (c_offset[1] + j) * n + (c_offset[0] + i)
            
        # M1 = (A00+A11)(B00+B11)
        add_term([(a_idx(0,0), 1), (a_idx(1,1), 1)],
                 [(b_idx(0,0), 1), (b_idx(1,1), 1)],
                 [(c_idx(0,0), 1), (c_idx(1,1), 1)])
                 
        # M2 = (A10+A11)B00
        add_term([(a_idx(1,0), 1), (a_idx(1,1), 1)],
                 [(b_idx(0,0), 1)],
                 [(c_idx(1,0), 1), (c_idx(1,1), -1)]) # Wait, C10 = M2+M4, C11 = M1-M2+M3+M6
        # Let's re-verify Strassen output mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M2:
        # U: A10, A11
        # V: B00
        # W: C10 (+1), C11 (-1)
        add_term([(a_idx(1,0), 1), (a_idx(1,1), 1)],
                 [(b_idx(0,0), 1)],
                 [(c_idx(1,0), 1), (c_idx(1,1), -1)])
                 
        # M3 = A00(B10-B11)
        add_term([(a_idx(0,0), 1)],
                 [(b_idx(1,0), 1), (b_idx(1,1), -1)],
                 [(c_idx(0,1), 1), (c_idx(1,1), 1)])
                 
        # M4 = A11(B01-B10)
        add_term([(a_idx(1,1), 1)],
                 [(b_idx(0,1), 1), (b_idx(1,0), -1)],
                 [(c_idx(0,0), 1), (c_idx(1,0), 1)])
                 
        # M5 = (A00+A01)B11
        add_term([(a_idx(0,0), 1), (a_idx(0,1), 1)],
                 [(b_idx(1,1), 1)],
                 [(c_idx(0,0), -1), (c_idx(0,1), 1)])
                 
        # M6 = (A10-A00)(B00+B01)
        add_term([(a_idx(1,0), 1), (a_idx(0,0), -1)],
                 [(b_idx(0,0), 1), (b_idx(0,1), 1)],
                 [(c_idx(1,1), 1)])
                 
        # M7 = (A01-A11)(B10+B11)
        add_term([(a_idx(0,1), 1), (a_idx(1,1), -1)],
                 [(b_idx(1,0), 1), (b_idx(1,1), 1)],
                 [(c_idx(0,0), 1)])
        
    # Term 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # A11 offset: (0,0), B11 offset: (0,0), C11 offset: (0,0)
    add_strassen(2, 2, 2, (0,0), (0,0), (0,0))
    
    # Term 2: A12 (2x3) * B21 (3x2) -> C11 (2x2)
    # A12 offset: (0,2), B21 offset: (2,0), C11 offset: (0,0)
    # Split A12 into A12a (2x1) and A12b (2x2)
    # Split B21 into B21a (1x2) and B21b (2x2)
    # A12a offset: (0,2), B21a offset: (2,0), C11 offset: (0,0)
    add_trivial(2, 1, 2, (0,2), (2,0), (0,0))
    # A12b offset: (0,3), B21b offset: (3,0), C11 offset: (0,0)
    add_strassen(2, 2, 2, (0,3), (3,0), (0,0))
    
    # Term 3: A11 (2x2) * B12 (2x3) -> C12 (2x3)
    # A11 offset: (0,0), B12 offset: (0,2), C12 offset: (0,2)
    # Split B12 into B12a (2x1) and B12b (2x2)
    # B12a offset: (0,2), B12b offset: (0,3)
    add_trivial(2, 2, 1, (0,0), (0,2), (0,2))
    add_strassen(2, 2, 2, (0,0), (0,3), (0,3))
    
    # Term 4: A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # A21 offset: (2,0), B11 offset: (0,0), C21 offset: (2,0)
    add_trivial(1, 2, 2, (2,0), (0,0), (2,0))
    
    # Term 5: A22 (1x3) * B21 (3x2) -> C21 (1x2)
    # A22 offset: (2,2), B21 offset: (2,0), C21 offset: (2,0)
    # Split A22 into A22a (1x1) and A22b (1x2)
    # Split B21 into B21a (1x2) and B21b (2x2)
    # A22a offset: (2,2), B21a offset: (2,0)
    add_trivial(1, 1, 2, (2,2), (2,0), (2,0))
    # A22b offset: (2,3), B21b offset: (3,0)
    add_trivial(1, 2, 2, (2,3), (3,0), (2,0))
    
    # Term 6: A21 (1x2) * B12 (2x3) -> C22 (1x3)
    # A21 offset: (2,0), B12 offset: (0,2), C22 offset: (2,2)
    # Split B12 into B12a (2x1) and B12b (2x2)
    # B12a offset: (0,2), B12b offset: (0,3)
    add_trivial(1, 2, 1, (2,0), (0,2), (2,2))
    add_trivial(1, 2, 2, (2,0), (0,3), (2,3))
    
    # Term 7: A22 (1x3) * B22 (3x3) -> C22 (1x3)
    # A22 offset: (2,2), B22 offset: (2,2), C22 offset: (2,2)
    # Split A22 into A22a (1x1) and A22b (1x2)
    # Split B22 into B22a (1x3) and B22b (2x3)
    # A22a offset: (2,2), B22a offset: (2,2)
    add_trivial(1, 1, 3, (2,2), (2,2), (2,2))
    # A22b offset: (2,3), B22b offset: (3,2)
    add_trivial(1, 2, 3, (2,3), (3,2), (2,2))
    
    rank = len(U_cols)
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
