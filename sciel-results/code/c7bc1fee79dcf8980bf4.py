"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def strassen_2x2(U, V, W, offset_u, offset_v, offset_w, base_col):
        """Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        U: list of lists (n*m rows, rank cols) - we'll append to it
        V: list of lists (m*p rows, rank cols)
        W: list of lists (p*n rows, rank cols)
        offset_u: row offset in U for this 2x2 block
        offset_v: row offset in V for this 2x2 block
        offset_w: row offset in W for this 2x2 block
        base_col: starting column index
        Returns: new base_col (advanced by 7)
        """
        # Strassen for 2x2 * 2x2:
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
        
        col = base_col
        # For each of the 7 Strassen terms, we need to express them as rank-1 contributions
        # Each term is a product of (sum/diff of A blocks) * (sum/diff of B blocks)
        # We need to expand these into elementary rank-1 tensors
        
        # Let's define the 7 Strassen matrices and their contributions
        # M1 = (A11+A22)*(B11+B22)
        # This expands to 4 rank-1 terms: A11*B11, A11*B22, A22*B11, A22*B22
        # But we want to use the Strassen structure to get rank 7 total for the 2x2*2x2
        
        # Actually, for the tensor decomposition, each Strassen "multiplication" M_i = X_i * Y_i
        # where X_i is a linear combination of A blocks and Y_i is a linear combination of B blocks.
        # Then C = sum of contributions from M_i.
        
        # Let me think of it differently. The Strassen algorithm computes:
        # M1 = (A11+A22) * (B11+B22)  -- this is a 2x2 matrix product
        # But in the tensor decomposition, we're decomposing the bilinear map.
        
        # For a 2x2 * 2x2 multiplication, the Strassen rank-7 decomposition is:
        # The 4 output entries C11, C12, C21, C22 are linear combinations of 7 products.
        # Each product M_i is of the form (linear combo of A entries) * (linear combo of B entries)
        
        # Standard Strassen for 2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_i contributes to U, V, W as follows:
        # U has rows indexed by (i,j) for A[i,j], V has rows indexed by (j,k) for B[j,k], W has rows indexed by (k,i) for C[k,i]
        
        # For M1 = (a11+a22)(b11+b22):
        # U contribution: +1 for (0,0), +1 for (1,1)
        # V contribution: +1 for (0,0), +1 for (1,1)
        # W contribution: +1 for (0,0) [C11], +1 for (1,1) [C22]... 
        # Wait, W[k*n+i, c] where C[k,i] is the output. For 2x2*2x2, n=2, p=2.
        # W index: k*n + i = k*2 + i
        # C11 is at W[0*2+0] = W[0]
        # C12 is at W[1*2+0] = W[2]  (k=1, i=0)
        # C21 is at W[0*2+1] = W[1]  (k=0, i=1)
        # C22 is at W[1*2+1] = W[3]  (k=1, i=1)
        
        # Let me build the 7 Strassen terms as rank-1 contributions to U, V, W
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1 = (a11+a22)(b11+b22)
            # U: a11 -> (0,0), a22 -> (1,1)
            # V: b11 -> (0,0), b22 -> (1,1)
            # W: C11 -> (0,0) i.e. W[0], C22 -> (1,1) i.e. W[3]
            # But wait, M1 contributes to both C11 and C22 with coefficient +1
            # So W: +1 at W[0] (C11), +1 at W[3] (C22)
            # But this is one rank-1 term? No, each M_i is ONE product of two linear forms.
            # M1 = (a11+a22) * (b11+b22) is a single bilinear term.
            # In the tensor decomposition, this contributes:
            # U: +1 for A[0,0] (index 0*2+0=0), +1 for A[1,1] (index 1*2+1=3)
            # V: +1 for B[0,0] (index 0*2+0=0), +1 for B[1,1] (index 1*2+1=3)
            # W: The product (a11+a22)(b11+b22) contributes to C11 and C22
            # C11 = M1 + M4 - M5 + M7, so M1 contributes +1 to C11
            # C22 = M1 - M2 + M3 + M6, so M1 contributes +1 to C22
            # W: +1 at W[0] (C[0,0]), +1 at W[3] (C[1,1])
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            
            # M2 = (a21+a22)b11
            # U: a21 -> (1,0) idx 2, a22 -> (1,1) idx 3
            # V: b11 -> (0,0) idx 0
            # W: C21 = M2 + M4, so +1 to C21 = W[1]
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
            
            # M3 = a11(b12-b22)
            # U: a11 -> (0,0) idx 0
            # V: b12 -> (0,1) idx 1, -b22 -> (1,1) idx 3
            # W: C12 = M3 + M5, C22 = ... + M3, so +1 to C12=W[2], +1 to C22=W[3]
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            
            # M4 = a22(b21-b11)
            # U: a22 -> (1,1) idx 3
            # V: b21 -> (1,0) idx 2, -b11 -> (0,0) idx 0
            # W: C11 = ... + M4 ..., C21 = ... + M4, so +1 to C11=W[0], +1 to C21=W[1]
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            
            # M5 = (a11+a12)b22
            # U: a11 -> (0,0) idx 0, a12 -> (0,1) idx 1
            # V: b22 -> (1,1) idx 3
            # W: C11 = ... - M5 ..., C12 = ... + M5, so -1 to C11=W[0], +1 to C12=W[2]
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            
            # M6 = (a21-a11)(b11+b12)
            # U: a21 -> (1,0) idx 2, -a11 -> (0,0) idx 0
            # V: b11 -> (0,0) idx 0, b12 -> (0,1) idx 1
            # W: C22 = ... + M6, so +1 to C22=W[3]
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            
            # M7 = (a12-a22)(b21+b22)
            # U: a12 -> (0,1) idx 1, -a22 -> (1,1) idx 3
            # V: b21 -> (1,0) idx 2, b22 -> (1,1) idx 3
            # W: C11 = ... + M7, so +1 to C11=W[0]
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for idx_u, val_u in enumerate(u_coeffs):
                if val_u != 0:
                    U[offset_u + idx_u][col] = val_u
            for idx_v, val_v in enumerate(v_coeffs):
                if val_v != 0:
                    V[offset_v + idx_v][col] = val_v
            for idx_w, val_w in enumerate(w_coeffs):
                if val_w != 0:
                    W[offset_w + idx_w][col] = val_w
            col += 1
        
        return col
    
    # For 4x4 * 4x4: use 2-level Strassen (rank 49)
    # For 4x4 * 4x2: use Strassen on 2x2 blocks (rank 28)
    # Total rank = 49 + 28 = 77
    
    rank = 77
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    V = [[0] * rank for _ in range(24)]  # 4*6 = 24 rows
    W = [[0] * rank for _ in range(24)]  # 6*4 = 24 rows
    
    col = 0
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B, producing columns 0-3 of C)
    # B is 4x6, so B[:, 0:4] is 4x4
    # A is 4x4
    # We partition A into 2x2 blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B1 = B[:, 0:4] partitioned as B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    
    # Apply Strassen at the 4x4 block level: 7 block multiplications of 2x2 blocks
    # Each block multiplication is 2x2 * 2x2, done via Strassen (rank 7)
    
    # The 7 Strassen block multiplications for 4x4*4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each M_i is a 2x2 * 2x2 product, decomposed via Strassen into 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # Define the 7 Strassen block-level terms with their U, V, W block coefficients
    # U blocks: A11 (rows 0-3), A12 (rows 4-7), A21 (rows 8-11), A22 (rows 12-15)
    # V blocks: B11 (rows 0-3), B12 (rows 4-7), B21 (rows 8-11), B22 (rows 12-15)
    # W blocks: C11 (rows 0-3), C12 (rows 4-7), C21 (rows 8-11), C22 (rows 12-15)
    # W index: k*n+i, for 4x4 output, k=0..3, i=0..3
    # C11: C[0:2,0:2] -> W[0:4]
    # C12: C[0:2,2:4] -> W[4:8]  (k=0,1; i=2,3 -> W[0*4+2]=W[2]? No...)
    
    # Wait, W[k*n+i] where n=4. For C[i,j] (i=row, j=col), W index = j*n + i
    # C11 = C[0:2, 0:2]: i=0,1; j=0,1 -> W[0*4+0]=W[0], W[0*4+1]=W[1], W[1*4+0]=W[4], W[1*4+1]=W[5]
    # C12 = C[0:2, 2:4]: i=0,1; j=2,3 -> W[2*4+0]=W[8], W[2*4+1]=W[9], W[3*4+0]=W[12], W[3*4+1]=W[13]
    # C21 = C[2:4, 0:2]: i=2,3; j=0,1 -> W[0*4+2]=W[2], W[0*4+3]=W[3], W[1*4+2]=W[6], W[1*4+3]=W[7]
    # C22 = C[2:4, 2:4]: i=2,3; j=2,3 -> W[2*4+2]=W[10], W[2*4+3]=W[11], W[3*4+2]=W[14], W[3*4+3]=W[15]
    
    # So W block offsets:
    # C11: W[0,1,4,5] -> not contiguous! This complicates things.
    
    # Let me reconsider. The W matrix has rows indexed by k*n+i where k is the column of C, i is the row of C.
    # For a 2x2 sub-block of C at position (block_row, block_col) in the 2x2 block grid:
    # block_row=0, block_col=0: C[0:2, 0:2], i=0,1, j=0,1 -> W[j*4+i] for j in {0,1}, i in {0,1}
    #   = W[0], W[1], W[4], W[5]
    
    # This is getting complex. Let me just directly construct the rank-1 terms for the full 4x4*4x4 case.
    
    # Alternative approach: directly enumerate all rank-1 terms for the Strassen decomposition.
    
    # Let me restart with a cleaner approach. I'll build the U, V, W matrices by adding contributions from each rank-1 term.
    
    # Clear and rebuild
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    col = 0
    
    # Helper: add a rank-1 term where U has entries at specified (row, value) pairs, etc.
    def add_term(U_rows, V_rows, W_rows, col_idx):
        """U_rows, V_rows, W_rows are dicts mapping row_index -> value"""
        for row, val in U_rows.items():
            if val != 0:
                U[row][col_idx] = val
        for row, val in V_rows.items():
            if val != 0:
                V[row][col_idx] = val
        for row, val in W_rows.items():
            if val != 0:
                W[row][col_idx] = val
    
    # Part 1: 4x4 * 4x4 via 2-level Strassen
    # A: 4x4, rows 0-15 in U (row i*4+j for A[i,j])
    # B[:, 0:4]: 4x4, rows 0-15 in V (row j*4+k for B[j,k] where k=0..3)
    # C[:, 0:4]: 4x4, rows in W (row k*4+i for C[i,k] where k=0..3, i=0..3)
    
    # 2-level Strassen: 7 block multiplications, each 2x2*2x2 via Strassen (7 terms each)
    
    # Block-level Strassen terms for 4x4*4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # A block offsets in U: A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    # B block offsets in V: B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
    # C block offsets in W:
    # C11 (C[0:2,0:2]): W indices for i=0,1, j=0,1: W[0*4+0]=0, W[0*4+1]=1, W[1*4+0]=4, W[1*4+1]=5
    # C12 (C[0:2,2:4]): i=0,1, j=2,3: W[2*4+0]=8, W[2*4+1]=9, W[3*4+0]=12, W[3*4+1]=13
    # C21 (C[2:4,0:2]): i=2,3, j=0,1: W[0*4+2]=2, W[0*4+3]=3, W[1*4+2]=6, W[1*4+3]=7
    # C22 (C[2:4,2:4]): i=2,3, j=2,3: W[2*4+2]=10, W[2*4+3]=11, W[3*4+2]=14, W[3*4+3]=15
    
    # For each block-level Strassen term M_i, we need to decompose it as a 2x2*2x2 product using Strassen.
    # The 2x2*2x2 Strassen gives 7 rank-1 terms.
    
    # Let me define a function that, given the U block offsets and V block offsets for a 2x2*2x2 product,
    # and the W indices for the 4 output entries, adds the 7 Strassen rank-1 terms.
    
    def add_strassen_2x2(U_off, V_off, W_indices, base_col):
        """Add 7 rank-1 Strassen terms for a 2x2*2x2 product.
        U_off: list of 4 row indices in U for A[0,0], A[0,1], A[1,0], A[1,1]
        V_off: list of 4 row indices in V for B[0,0], B[0,1], B[1,0], B[1,1]
        W_indices: list of 4 row indices in W for C[0,0], C[0,1], C[1,0], C[1,1]
        base_col: starting column
        Returns: new base_col
        """
        col = base_col
        # 7 Strassen terms for 2x2*2x2
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: [A00, A01, A10, A11] coefficients
        # V_coeffs: [B00, B01, B10, B11] coefficients
        # W_coeffs: [C00, C01, C10, C11] coefficients
        
        terms = [
            # M1 = (A00+A11)(B00+B11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2 = (A10+A11)B00
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            # M3 = A00(B01-B11)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # M4 = A11(B10-B00)
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            # M5 = (A00+A01)B11
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6 = (A10-A00)(B00+B01)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7 = (A01-A11)(B10+B11)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_c, v_c, w_c in terms:
            for idx, val in enumerate(u_c):
                if val != 0:
                    U[U_off[idx]][col] = val
            for idx, val in enumerate(v_c):
                if val != 0:
                    V[V_off[idx]][col] = val
            for idx, val in enumerate(w_c):
                if val != 0:
                    W[W_indices[idx]][col] = val
            col += 1
        return col
    
    # Now define the 7 block-level Strassen terms for 4x4*4x4
    # Each involves specific A blocks and B blocks
    
    # A block row indices in U:
    # A11: [0,1,4,5]  (A[0,0], A[0,1], A[1,0], A[1,1])
    # A12: [2,3,6,7]  (A[0,2], A[0,3], A[1,2], A[1,3])
    # A21: [8,9,12,13] (A[2,0], A[2,1], A[3,0], A[3,1])
    # A22: [10,11,14,15] (A[2,2], A[2,3], A[3,2], A[3,3])
    
    # B block row indices in V (for B[:, 0:4]):
    # B11: [0,1,4,5]  (B[0,0], B[0,1], B[1,0], B[1,1])
    # B12: [2,3,6,7]  (B[0,2], B[0,3], B[1,2], B[1,3])
    # B21: [8,9,12,13] (B[2,0], B[2,1], B[3,0], B[3,1])
    # B22: [10,11,14,15] (B[2,2], B[2,3], B[3,2], B[3,3])
    
    # W indices for C blocks (C[:, 0:4]):
    # C11: C[0:2,0:2] -> i=0,1, j=0,1: W[j*4+i]: W[0], W[1], W[4], W[5]
    #   C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[4], C[1,1]=W[5]
    # C12: C[0:2,2:4] -> i=0,1, j=2,3: W[8], W[9], W[12], W[13]
    #   C[0,2]=W[8], C[0,3]=W[9], C[1,2]=W[12], C[1,3]=W[13]
    # C21: C[2:4,0:2] -> i=2,3, j=0,1: W[2], W[3], W[6], W[7]
    #   C[2,0]=W[2], C[2,1]=W[3], C[3,0]=W[6], C[3,1]=W[7]
    # C22: C[2:4,2:4] -> i=2,3, j=2,3: W[10], W[11], W[14], W[15]
    #   C[2,2]=W[10], C[2,3]=W[11], C[3,2]=W[14], C[3,3]=W[15]
    
    A11_U = [0, 1, 4, 5]
    A12_U = [2, 3, 6, 7]
    A21_U = [8, 9, 12, 13]
    A22_U = [10, 11, 14, 15]
    
    B11_V = [0, 1, 4, 5]
    B12_V = [2, 3, 6, 7]
    B21_V = [8, 9, 12, 13]
    B22_V = [10, 11, 14, 15]
    
    C11_W = [0, 1, 4, 5]
    C12_W = [8, 9, 12, 13]
    C21_W = [2, 3, 6, 7]
    C22_W = [10, 11, 14, 15]
    
    # 7 block-level Strassen terms:
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11 and C22
    # M2 = (A21+A22)*B11 -> contributes to C21
    # M3 = A11*(B12-B22) -> contributes to C12 and C22
    # M4 = A22*(B21-B11) -> contributes to C11 and C21
    # M5 = (A11+A12)*B22 -> contributes to C11 and C12
    # M6 = (A21-A11)*(B11+B12) -> contributes to C22
    # M7 = (A12-A22)*(B21+B22) -> contributes to C11
    
    # For each, we need to decompose the 2x2*2x2 product using Strassen.
    # The challenge: the A and B blocks may be sums/differences.
    
    # M1 = (A11+A22)*(B11+B22)
    # This is a 2x2*2x2 product where the "A" is A11+A22 and "B" is B11+B22.
    # In the tensor decomposition, this means the U coefficients are the sum of A11 and A22 entries,
    # and V coefficients are the sum of B11 and B22 entries.
    # But we can't just add them because each rank-1 term must have specific U, V, W vectors.
    
    # Actually, (A11+A22)*(B11+B22) is NOT a single 2x2*2x2 product in the usual sense.
    # It's the product of two 2x2 matrices: X = A11+A22 and Y = B11+B22.
    # So we apply Strassen to X*Y, where X and Y are 2x2 matrices.
    # The U vectors will have entries corresponding to X's entries (which are sums of A11 and A22 entries),
    # and V vectors will have entries corresponding to Y's entries (sums of B11 and B22 entries).
    
    # So for M1, the 7 Strassen terms will have U coefficients that are combinations of A11 and A22 entries,
    # and V coefficients that are combinations of B11 and B22 entries.
    
    # Let me define a more general function:
    
    def add_strassen_product(U_blocks, V_blocks, W_blocks, W_coeffs_per_output, base_col):
        """Add Strassen decomposition for X*Y where X is sum of U_blocks, Y is sum of V_blocks.
        U_blocks: list of (block_U_indices, coeff) pairs
        V_blocks: list of (block_V_indices, coeff) pairs
        W_blocks: dict mapping output_block_name -> W_indices
        W_coeffs_per_output: dict mapping output_block_name -> coeff
        """
        pass  # This is getting too complex. Let me take a different approach.
    
    # Let me just directly enumerate all 49 rank-1 terms for the 4x4*4x4 part.
    # I'll compute them systematically.
    
    # For each of the 7 block-level Strassen terms, I'll compute the 7 sub-Strassen terms.
    
    # Let me define the block-level terms more carefully:
    
    # M1: X = A11 + A22, Y = B11 + B22, contributes +1 to C11, +1 to C22
    # M2: X = A21 + A22, Y = B11, contributes +1 to C21
    # M3: X = A11, Y = B12 - B22, contributes +1 to C12, +1 to C22
    # M4: X = A22, Y = B21 - B11, contributes +1 to C11, +1 to C21
    # M5: X = A11 + A12, Y = B22, contributes -1 to C11, +1 to C12
    # M6: X = A21 - A11, Y = B11 + B12, contributes +1 to C22
    # M7: X = A12 - A22, Y = B21 + B22, contributes +1 to C11
    
    # For each, I'll call add_strassen_2x2 with appropriate U, V, W indices.
    # But the U and V blocks are sums, so I need to handle that.
    
    # Let me create a helper that takes lists of (U_index, coeff) for the 4 A entries,
    # (V_index, coeff) for the 4 B entries, and (W_index, coeff) for the 4 C entries.
    
    def add_strassen_expanded(A_coeffs, B_coeffs, C_coeffs, base_col):
        """A_coeffs: list of 4 (row, coeff) for A00, A01, A10, A11
        B_coeffs: list of 4 (row, coeff) for B00, B01, B10, B11
        C_coeffs: list of 4 (row, coeff) for C00, C01, C10, C11
        """
        col = base_col
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_c, v_c, w_c in terms:
            for idx in range(4):
                u_row, u_base = A_coeffs[idx]
                u_val = u_c[idx] * u_base
                if u_val != 0:
                    U[u_row][col] = u_val
                v_row, v_base = B_coeffs[idx]
                v_val = v_c[idx] * v_base
                if v_val != 0:
                    V[v_row][col] = v_val
                w_row, w_base = C_coeffs[idx]
                w_val = w_c[idx] * w_base
                if w_val != 0:
                    W[w_row][col] = w_val
            col += 1
        return col
    
    # Now define the 7 block-level terms:
    
    # M1: X = A11 + A22, Y = B11 + B22, C: +1*C11, +1*C22
    # A_coeffs: A00=(A11[0],1), A01=(A11[1],1), A10=(A11[2],1), A11=(A11[3],1)
    #           plus A22 contributions: A00 also gets A22[0], etc.
    # So A00 total: (A11_U[0], 1) and (A22_U[0], 1) -> but we need a single (row, coeff) per position.
    # This doesn't work with my helper. Let me modify.
    
    # I need a more flexible approach. Let me just directly add the contributions.
    
    def add_strassen_block(A_blocks, B_blocks, C_blocks, base_col):
        """A_blocks: list of 4 entries, each is a list of (U_row, coeff)
        B_blocks: list of 4 entries, each is a list of (V_row, coeff)
        C_blocks: list of 4 entries, each is a list of (W_row, coeff)
        """
        col = base_col
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_c, v_c, w_c in terms:
            for idx in range(4):
                for u_row, u_base in A_blocks[idx]:
                    u_val = u_c[idx] * u_base
                    if u_val != 0:
                        U[u_row][col] += u_val
                for v_row, v_base in B_blocks[idx]:
                    v_val = v_c[idx] * v_base
                    if v_val != 0:
                        V[v_row][col] += v_val
                for w_row, w_base in C_blocks[idx]:
                    w_val = w_c[idx] * w_base
                    if w_val != 0:
                        W[w_row][col] += w_val
            col += 1
        return col
    
    # Define block contributions:
    # A11 blocks: [[(A11_U[0],1)], [(A11_U[1],1)], [(A11_U[2],1)], [(A11_U[3],1)]]
    # A22 blocks: [[(A22_U[0],1)], [(A22_U[1],1)], [(A22_U[2],1)], [(A22_U[3],1)]]
    # etc.
    
    def block_from_indices(indices):
        return [[(indices[i], 1)] for i in range(4)]
    
    def neg_block_from_indices(indices):
        return [[(indices[i], -1)] for i in range(4)]
    
    A11_b = block_from_indices(A11_U)
    A12_b = block_from_indices(A12_U)
    A21_b = block_from_indices(A21_U)
    A22_b = block_from_indices(A22_U)
    
    B11_b = block_from_indices(B11_V)
    B12_b = block_from_indices(B12_V)
    B21_b = block_from_indices(B21_V)
    B22_b = block_from_indices(B22_V)
    
    C11_b = block_from_indices(C11_W)
    C12_b = block_from_indices(C12_W)
    C21_b = block_from_indices(C21_W)
    C22_b = block_from_indices(C22_W)
    
    # M1: X = A11 + A22, Y = B11 + B22, C: +C11, +C22
    A_M1 = [[A11_b[i][0][0] and A11_b[i][0] or [] + A22_b[i][0] for i in range(4)]]  # This is getting messy
    
    # Let me just concatenate the block lists:
    def combine_blocks(b1, b2, sign1=1, sign2=1):
        result = []
        for i in range(4):
            result.append([(b1[i][0][0], sign1 * b1[i][0][1]), (b2[i][0][0], sign2 * b2[i][0][1])])
        return result
    
    A_M1 = combine_blocks(A11_b, A22_b, 1, 1)
    B_M1 = combine_blocks(B11_b, B22_b, 1, 1)
    C_M1 = combine_blocks(C11_b, C22_b, 1, 1)
    col = add_strassen_block(A_M1, B_M1, C_M1, col)
    
    # M2: X = A21 + A22, Y = B11, C: +C21
    A_M2 = combine_blocks(A21_b, A22_b, 1, 1)
    B_M2 = B11_b
    C_M2 = C21_b
    col = add_strassen_block(A_M2, B_M2, C_M2, col)
    
    # M3: X = A11, Y = B12 - B22, C: +C12, +C22
    A_M3 = A11_b
    B_M3 = combine_blocks(B12_b, B22_b, 1, -1)
    C_M3 = combine_blocks(C12_b, C22_b, 1, 1)
    col = add_strassen_block(A_M3, B_M3, C_M3, col)
    
    # M4: X = A22, Y = B21 - B11, C: +C11, +C21
    A_M4 = A22_b
    B_M4 = combine_blocks(B21_b, B11_b, 1, -1)
    C_M4 = combine_blocks(C11_b, C21_b, 1, 1)
    col = add_strassen_block(A_M4, B_M4, C_M4, col)
    
    # M5: X = A11 + A12, Y = B22, C: -C11, +C12
    A_M5 = combine_blocks(A11_b, A12_b, 1, 1)
    B_M5 = B22_b
    C_M5 = combine_blocks(C11_b, C12_b, -1, 1)
    col = add_strassen_block(A_M5, B_M5, C_M5, col)
    
    # M6: X = A21 - A11, Y = B11 + B12, C: +C22
    A_M6 = combine_blocks(A21_b, A11_b, 1, -1)
    B_M6 = combine_blocks(B11_b, B12_b, 1, 1)
    C_M6 = C22_b
    col = add_strassen_block(A_M6, B_M6, C_M6, col)
    
    # M7: X = A12 - A22, Y = B21 + B22, C: +C11
    A_M7 = combine_blocks(A12_b, A22_b, 1, -1)
    B_M7 = combine_blocks(B21_b, B22_b, 1, 1)
    C_M7 = C11_b
    col = add_strassen_block(A_M7, B_M7, C_M7, col)
    
    # col should now be 49
    assert col == 49, f"Expected col=49, got {col}"
    
    # Part 2: 4x4 * 4x2 (columns 4-5 of B, producing columns 4-5 of C)
    # B[:, 4:6] is 4x2
    # A is 4x4
    # C[:, 4:6] is 4x2
    
    # Partition A into 2x2 blocks: A11, A12, A21, A22 (same as before)
    # B[:, 4:6] partitioned as: B_top = B[0:2, 4:6] (2x2), B_bot = B[2:4, 4:6] (2x2)
    # C[:, 4:6] partitioned as: C_top = C[0:2, 4:6] (2x2), C_bot = C[2:4, 4:6] (2x2)
    
    # C_top = A11 * B_top + A12 * B_bot
    # C_bot = A21 * B_top + A22 * B_bot
    
    # We need 4 products: A11*B_top, A12*B_bot, A21*B_top, A22*B_bot
    # Each is 2x2*2x2, using Strassen (rank 7 each)
    # Total: 4 * 7 = 28
    
    # V indices for B[:, 4:6]:
    # B_top = B[0:2, 4:6]: B[0,4], B[0,5], B[1,4], B[1,5]
    #   V index: j*6+k: B[0,4]=V[0*6+4]=V[4], B[0,5]=V[5], B[1,4]=V[1*6+4]=V[10], B[1,5]=V[11]
    # B_bot = B[2:4, 4:6]: B[2,4], B[2,5], B[3,4], B[3,5]
    #   V index: B[2,4]=V[2*6+4]=V[16], B[2,5]=V[17], B[3,4]=V[3*6+4]=V[22], B[3,5]=V[23]
    
    # W indices for C[:, 4:6]:
    # C_top = C[0:2, 4:6]: C[0,4], C[0,5], C[1,4], C[1,5]
    #   W index: k*4+i: C[0,4]=W[4*4+0]=W[16], C[0,5]=W[5*4+0]=W[20], C[1,4]=W[4*4+1]=W[17], C[1,5]=W[5*4+1]=W[21]
    # C_bot = C[2:4, 4:6]: C[2,4], C[2,5], C[3,4], C[3,5]
    #   W index: C[2,4]=W[4*4+2]=W[18], C[2,5]=W[5*4+2]=W[22], C[3,4]=W[4*4+3]=W[19], C[3,5]=W[5*4+3]=W[23]
    
    B_top_V = [4, 5, 10, 11]
    B_bot_V = [16, 17, 22, 23]
    
    C_top_W = [16, 20, 17, 21]
    C_bot_W = [18, 22, 19, 23]
    
    B_top_b = block_from_indices(B_top_V)
    B_bot_b = block_from_indices(B_bot_V)
    C_top_b = block_from_indices(C_top_W)
    C_bot_b = block_from_indices(C_bot_W)
    
    # A11 * B_top -> contributes to C_top
    col = add_strassen_block(A11_b, B_top_b, C_top_b, col)
    
    # A12 * B_bot -> contributes to C_top
    col = add_strassen_block(A12_b, B_bot_b, C_top_b, col)
    
    # A21 * B_top -> contributes to C_bot
    col = add_strassen_block(A21_b, B_top_b, C_bot_b, col)
    
    # A22 * B_bot -> contributes to C_bot
    col = add_strassen_block(A22_b, B_bot_b, C_bot_b, col)
    
    assert col == 77, f"Expected col=77, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

