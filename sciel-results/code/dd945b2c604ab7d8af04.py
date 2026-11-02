"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid Strassen/Schoolbook decomposition for 3x4 * 4x7 matrix multiplication.
    # We split the 3x4 matrix A into top 2x4 block A1 and bottom 1x4 block A2.
    # We split the 4x7 matrix B into left 4x4 block B1 and right 4x3 block B2.
    # 
    # The top 2x4 * 4x4 part is decomposed using Strassen (2x2 * 2x2 blocks).
    # The remaining parts use schoolbook decomposition.
    # 
    # Total rank = 7*4 + 1*4 + 2*7 + 2*3 = 28 + 4 + 14 + 6 = 52? No, let me recalculate.
    # 
    # Actually, let's be more careful. The dimensions are n=3, m=4, p=7.
    # A is 3x4, B is 4x7, C is 3x7.
    # 
    # Let's split A into A_top (2x4) and A_bottom (1x4).
    # Let's split B into B_left (4x4) and B_right (4x3).
    # 
    # C = A * B = [A_top; A_bottom] * [B_left | B_right]
    #       = [A_top * B_left | A_top * B_right]
    #         [A_bottom * B_left | A_bottom * B_right]
    # 
    # We need to decompose:
    # 1. A_top * B_left: 2x4 * 4x4 -> 2x4
    # 2. A_top * B_right: 2x4 * 4x3 -> 2x3
    # 3. A_bottom * B_left: 1x4 * 4x4 -> 1x4
    # 4. A_bottom * B_right: 1x4 * 4x3 -> 1x3
    # 
    # For A_top * B_left (2x4 * 4x4):
    # Split A_top into 2x2 blocks: A_top = [A11 A12] where A11 is 2x2, A12 is 2x2.
    # Split B_left into 2x2 blocks: B_left = [[B11 B12], [B21 B22]] (4x4 = 2x(2x2)).
    # Wait, 2x4 * 4x4. We can view this as 2x(2x2) * (2x2)x(2x2).
    # Actually, let's just use schoolbook for simplicity first, then optimize.
    #
    # Schoolbook rank for 2x4 * 4x4 is 2*4*4 = 32.
    # Schoolbook rank for 2x4 * 4x3 is 2*4*3 = 24.
    # Schoolbook rank for 1x4 * 4x4 is 1*4*4 = 16.
    # Schoolbook rank for 1x4 * 4x3 is 1*4*3 = 12.
    # Total schoolbook rank = 32+24+16+12 = 84. This is the same as the full schoolbook.
    #
    # To improve, we need to use Strassen on the 2x4 * 4x4 part.
    # Let's write a general function to build a Strassen decomposition for 2x2 * 2x2.
    # And use it on the 2x4 * 4x4 block by splitting into four 2x2 * 2x2 sub-problems?
    # No, 2x4 * 4x4 can be seen as four 2x2 * 2x2 multiplications? No.
    # A_top is 2x4. B_left is 4x4.
    # C1 = A_top * B_left is 2x4.
    # We can split A_top into two 2x2 blocks: A_top = [A_L | A_R] (2x2 each).
    # We can split B_left into four 2x2 blocks: B_left = [[B11, B12], [B21, B22]].
    # Then C1 = [A_L B11 + A_R B21 | A_L B12 + A_R B22].
    # This requires 4 Strassen decompositions of 2x2 * 2x2.
    # Rank = 4 * 7 = 28.
    #
    # Similarly for A_top * B_right (2x4 * 4x3):
    # B_right is 4x3. We can't split it into 2x2 blocks cleanly for Strassen.
    # We can do schoolbook: 2*4*3 = 24.
    # Or we can split B_right into top 2x3 and bottom 2x3.
    # Then A_top * B_right = A_L * B_top + A_R * B_bottom.
    # Each is 2x2 * 2x3. Schoolbook rank for 2x2 * 2x3 is 2*2*3 = 12.
    # Total = 12 + 12 = 24. Same as before.
    #
    # For A_bottom * B_left (1x4 * 4x4):
    # Schoolbook: 1*4*4 = 16.
    # Can we do better? 1x4 * 4x4. This is a row vector times a matrix.
    # It's a sum of 4 scalar multiplications. No known better decomposition.
    #
    # For A_bottom * B_right (1x4 * 4x3):
    # Schoolbook: 1*4*3 = 12.
    #
    # Total rank = 28 (Strassen for 2x4*4x4) + 24 (schoolbook for 2x4*4x3) + 16 + 12 = 80.
    # This is better than 84.
    #
    # Let's implement this.

    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll build the U, V, W matrices as lists of lists.
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    U = []
    V = []
    W = []
    
    def add_decomp(U_sub, V_sub, W_sub, row_offset_U, row_offset_V, row_offset_W):
        """Add a sub-decomposition to the global matrices."""
        nonlocal U, V, W
        for c in range(len(U_sub[0])):
            # U
            U.append([U_sub[r][c] for r in range(len(U_sub))])
            # We need to pad with zeros for the other rows
            # Actually, it's easier to build the full matrices at once.
            pass
    
    # Let's build the full matrices directly.
    
    # Sub-decomposition 1: A_top * B_left using 4 Strassen blocks.
    # A_top is rows 0,1 of A (indices 0..3 in U, since U is n*m = 12, row i*m+j).
    # A_top rows: i=0,1. j=0..3. So U rows: 0*4+0=0, 1, 2, 3 for i=0; 4,5,6,7 for i=1.
    # Wait, U index is i*m+j. So for i=0, j=0..3: 0,1,2,3. For i=1, j=0..3: 4,5,6,7.
    # A_bottom is i=2: j=0..3: 8,9,10,11.
    #
    # B_left is columns 0..3 of B. V index is j*p+k. For j=0..3, k=0..3.
    # V rows: 0*7+0=0, 1, 2, 3, 4 (j=0, k=0..3), 7,8,9,10 (j=1, k=0..3), etc.
    #
    # C1 = A_top * B_left is 2x4. Output indices in W: k*n+i.
    # For i=0,1 and k=0..3.
    # W rows: 0*3+0=0, 1, 2 (k=0, i=0,1,2? No, k*n+i. For k=0: i=0,1,2 -> 0,1,2. k=1: 3,4,5. etc.)
    # We only care about i=0,1 for C1. So W rows 0,1 (k=0), 3,4 (k=1), 6,7 (k=2), 9,10 (k=3).
    
    # Let's define a helper to add a 2x2 * 2x2 Strassen decomposition.
    
    def strassen_2x2(U_rows, V_rows, W_rows, U_cols, V_cols, W_cols):
        """
        Add a Strassen decomposition for 2x2 * 2x2.
        U_rows: list of 4 row indices in U (for the 2x2 block of A).
        V_rows: list of 4 row indices in V (for the 2x2 block of B).
        W_rows: list of 4 row indices in W (for the 2x2 block of C).
        U_cols: list of 2 column indices in A (local).
        V_cols: list of 2 column indices in B (local).
        W_cols: list of 2 column indices in C (local).
        Returns the rank added.
        """
        nonlocal U, V, W
        # This is complex. Let's just build the full matrices in one go.
        pass

    # Given the complexity, let's just use schoolbook but with a different grouping?
    # No, the directive says to use a new approach. Strassen is the way.
    
    # Let's manually build the 4 Strassen decompositions for A_top * B_left.
    # A_top = [A11 A12] where A11 is rows 0,1 cols 0,1 of A. A12 is rows 0,1 cols 2,3.
    # B_left = [[B11 B12], [B21 B22]] where B11 is rows 0,1 cols 0,1 of B. etc.
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For our 2x4 * 4x4, we have four such blocks.
    # Block 1: A11 * B11. A11 is A[0:2, 0:2], B11 is B[0:2, 0:2].
    # Block 2: A12 * B12. A12 is A[0:2, 2:4], B12 is B[0:2, 2:4].
    # Block 3: A11 * B21. A11 is A[0:2, 0:2], B21 is B[2:4, 0:2].
    # Block 4: A12 * B22. A12 is A[0:2, 2:4], B22 is B[2:4, 2:4].
    
    # We'll add these 4 Strassen decompositions.
    
    rank = 0
    U_mat = [[0] * 0 for _ in range(n * m)]  # Will resize
    V_mat = [[0] * 0 for _ in range(m * p)]
    W_mat = [[0] * 0 for _ in range(p * n)]
    
    # Let's just build the lists of entries for each column.
    U_entries = []
    V_entries = []
    W_entries = []
    
    def add_column(u_vals, v_vals, w_vals):
        nonlocal U_entries, V_entries, W_entries
        U_entries.append(u_vals)
        V_entries.append(v_vals)
        W_entries.append(w_vals)
    
    # Helper to create a column with specific non-zero entries.
    def make_col(size, entries):
        col = [0] * size
        for idx, val in entries.items():
            col[idx] = val
        return col
    
    # Strassen 2x2 * 2x2.
    # A is 2x2, B is 2x2, C is 2x2.
    # We need to map local indices to global.
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        a_rows: global row indices for the 2x2 block of A (2 indices).
        a_cols: global col indices for the 2x2 block of A (2 indices).
        b_rows: global row indices for the 2x2 block of B (2 indices).
        b_cols: global col indices for the 2x2 block of B (2 indices).
        c_rows: global row indices for the 2x2 block of C (2 indices).
        c_cols: global col indices for the 2x2 block of C (2 indices).
        """
        # A11 is (a_rows[0], a_cols[0]), A12 is (a_rows[0], a_cols[1])
        # A21 is (a_rows[1], a_cols[0]), A22 is (a_rows[1], a_cols[1])
        # Similarly for B and C.
        
        # U index: i*m + j. Here i is in a_rows, j is in a_cols.
        # V index: j*p + k. Here j is in b_rows, k is in b_cols.
        # W index: k*n + i. Here k is in c_cols, i is in c_rows.
        
        # Let's denote the global indices.
        # A11_U = a_rows[0]*m + a_cols[0]
        # A12_U = a_rows[0]*m + a_cols[1]
        # A21_U = a_rows[1]*m + a_cols[0]
        # A22_U = a_rows[1]*m + a_cols[1]
        
        # B11_V = b_rows[0]*p + b_cols[0]
        # B12_V = b_rows[0]*p + b_cols[1]
        # B21_V = b_rows[1]*p + b_cols[0]
        # B22_V = b_rows[1]*p + b_cols[1]
        
        # C11_W = c_cols[0]*n + c_rows[0]
        # C12_W = c_cols[1]*n + c_rows[0]
        # C21_W = c_cols[0]*n + c_rows[1]
        # C22_W = c_cols[1]*n + c_rows[1]
        
        # M1 = (A11 + A22)(B11 + B22)
        u1 = make_col(n*m, {a_rows[0]*m + a_cols[0]: 1, a_rows[1]*m + a_cols[1]: 1})
        v1 = make_col(m*p, {b_rows[0]*p + b_cols[0]: 1, b_rows[1]*p + b_cols[1]: 1})
        w1 = make_col(p*n, {c_cols[0]*n + c_rows[0]: 1, c_cols[0]*n + c_rows[1]: -1, c_cols[1]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: 1})
        add_column(u1, v1, w1)
        
        # M2 = (A21 + A22)B11
        u2 = make_col(n*m, {a_rows[1]*m + a_cols[0]: 1, a_rows[1]*m + a_cols[1]: 1})
        v2 = make_col(m*p, {b_rows[0]*p + b_cols[0]: 1})
        w2 = make_col(p*n, {c_cols[0]*n + c_rows[1]: 1, c_cols[1]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: -1})
        add_column(u2, v2, w2)
        
        # M3 = A11(B12 - B22)
        u3 = make_col(n*m, {a_rows[0]*m + a_cols[0]: 1})
        v3 = make_col(m*p, {b_rows[0]*p + b_cols[1]: 1, b_rows[1]*p + b_cols[1]: -1})
        w3 = make_col(p*n, {c_cols[0]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: 1})
        add_column(u3, v3, w3)
        
        # M4 = A22(B21 - B11)
        u4 = make_col(n*m, {a_rows[1]*m + a_cols[1]: 1})
        v4 = make_col(m*p, {b_rows[1]*p + b_cols[0]: 1, b_rows[0]*p + b_cols[0]: -1})
        w4 = make_col(p*n, {c_cols[0]*n + c_rows[1]: 1, c_cols[1]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: -1})
        add_column(u4, v4, w4)
        
        # M5 = (A11 + A12)B22
        u5 = make_col(n*m, {a_rows[0]*m + a_cols[0]: 1, a_rows[0]*m + a_cols[1]: 1})
        v5 = make_col(m*p, {b_rows[1]*p + b_cols[1]: 1})
        w5 = make_col(p*n, {c_cols[0]*n + c_rows[0]: -1, c_cols[1]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: 1})
        add_column(u5, v5, w5)
        
        # M6 = (A21 - A11)(B11 + B12)
        u6 = make_col(n*m, {a_rows[1]*m + a_cols[0]: 1, a_rows[0]*m + a_cols[0]: -1})
        v6 = make_col(m*p, {b_rows[0]*p + b_cols[0]: 1, b_rows[0]*p + b_cols[1]: 1})
        w6 = make_col(p*n, {c_cols[0]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: 1})
        add_column(u6, v6, w6)
        
        # M7 = (A12 - A22)(B21 + B22)
        u7 = make_col(n*m, {a_rows[0]*m + a_cols[1]: 1, a_rows[1]*m + a_cols[1]: -1})
        v7 = make_col(m*p, {b_rows[1]*p + b_cols[0]: 1, b_rows[1]*p + b_cols[1]: 1})
        w7 = make_col(p*n, {c_cols[0]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: -1})
        add_column(u7, v7, w7)
    
    # Add the 4 Strassen blocks for A_top * B_left.
    # A_top is i=0,1. B_left is j=0..3, k=0..3.
    # Block 1: A11 (i=0,1, j=0,1) * B11 (j=0,1, k=0,1) -> C11 (i=0,1, k=0,1)
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A12 (i=0,1, j=2,3) * B12 (j=0,1, k=2,3) -> C12 (i=0,1, k=2,3)
    add_strassen([0, 1], [2, 3], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 3: A11 (i=0,1, j=0,1) * B21 (j=2,3, k=0,1) -> C21 (i=0,1, k=0,1)
    add_strassen([0, 1], [0, 1], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 4: A12 (i=0,1, j=2,3) * B22 (j=2,3, k=2,3) -> C22 (i=0,1, k=2,3)
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # Now schoolbook for the remaining parts.
    # A_top * B_right: i=0,1, j=0..3, k=4..6.
    for i in range(2):
        for j in range(4):
            for k in range(4, 7):
                u = make_col(n*m, {i*m + j: 1})
                v = make_col(m*p, {j*p + k: 1})
                w = make_col(p*n, {k*n + i: 1})
                add_column(u, v, w)
    
    # A_bottom * B_left: i=2, j=0..3, k=0..3.
    for j in range(4):
        for k in range(4):
            u = make_col(n*m, {2*m + j: 1})
            v = make_col(m*p, {j*p + k: 1})
            w = make_col(p*n, {k*n + 2: 1})
            add_column(u, v, w)
    
    # A_bottom * B_right: i=2, j=0..3, k=4..6.
    for j in range(4):
        for k in range(4, 7):
            u = make_col(n*m, {2*m + j: 1})
            v = make_col(m*p, {j*p + k: 1})
            w = make_col(p*n, {k*n + 2: 1})
            add_column(u, v, w)
    
    # Build the final matrices.
    rank = len(U_entries)
    U_mat = [U_entries[c] for c in range(rank)]
    # Transpose to (n*m, rank)
    U_mat = [[U_entries[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_entries[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_entries[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
