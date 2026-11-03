"""Direct-sum decomposition: two independent 4x4x4 multiplications via Strassen."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # We decompose 4x4 x 4x8 -> 4x8 as the direct sum of two 4x4 x 4x4 multiplications.
    # For each 4x4x4 block, we use Strassen's algorithm (rank 7).
    # Total rank = 7 + 7 = 14.
    #
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p:
    # T[i, j, k] = 1 if the (i,k) element of C depends on A[i,j] * B[j,k]
    # In our indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # where W[k*n + i, r] is the coefficient for C[i, k] from the r-th rank-1 term.
    #
    # For a 4x4x4 multiplication (n=m=p=4), Strassen's 7 rank-1 terms are:
    # We partition 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen's formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M5
    #
    # For the first 4x4x4 block (columns 0-3 of the 4x8 output):
    # A is 4x4 (indices 0-3 in both dims), B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    #
    # For the second 4x4x4 block (columns 4-7 of the 4x8 output):
    # A is the same 4x4, B is 4x4 (cols 4-7), C is 4x4 (cols 4-7)
    #
    # Let's build the rank-14 decomposition.
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 16 x 14
    V = [[0] * rank for _ in range(m * p)]  # 32 x 14
    W = [[0] * rank for _ in range(p * n)]  # 32 x 14
    
    # Strassen's 7 terms for a 2x2 block multiplication
    # Each term is a rank-1 tensor: U_col * V_col * W_col
    # For a 4x4 matrix split into 2x2 blocks of size 2x2:
    # A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    # A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    
    # Define the 7 Strassen terms for one 4x4x4 block
    # Each term contributes to U (16 entries), V (16 entries for this block), W (16 entries for this block)
    
    # Helper: for a 4x4 matrix, the 2x2 block (br, bc) has rows br*2..br*2+1, cols bc*2..bc*2+1
    # U index: i*m + j = i*4 + j (for 4x4 A)
    # V index: j*p + k -> for first block: j*4 + k (k=0..3), for second block: j*4 + (k+4)
    # W index: k*n + i -> for first block: k*4 + i (k=0..3), for second block: (k+4)*4 + i
    
    def add_strassen_block(v_offset, w_offset, col_base):
        """Add Strassen's 7 rank-1 terms for a 4x4x4 block.
        v_offset: offset in V's column index (0 for first block, 4 for second)
        w_offset: offset in W's row index (0 for first block, 4 for second)
        col_base: starting column in the rank dimension
        """
        # The 7 Strassen products, each defined by:
        # (A_factor, B_factor) where each factor is a linear combination of 2x2 blocks
        # We need to express each as: sum of A[i,j]*coeff * B[j,k]*coeff
        
        # For each Strassen term, we have:
        # A_part = combination of A blocks, B_part = combination of B blocks
        # The rank-1 term is: (A_part flattened) outer (B_part flattened) outer (C contribution)
        
        # Let's define the 7 terms explicitly.
        # Block indices: 00, 01, 10, 11 for (row_block, col_block)
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C contributions:
        # C00 += M1 - M4 + M6 - M7
        # C01 += M3 + M5
        # C10 += M2 + M4
        # C11 += M1 - M3 + M2 - M5
        
        # For each term t (0..6), we define:
        # U_contrib: 16-vector (for the 4x4 A matrix)
        # V_contrib: 16-vector (for the 4x4 B matrix, with offset)
        # W_contrib: 16-vector (for the 4x4 C matrix, with offset)
        
        # A block (br, bc) occupies U indices: (br*2 + di)*4 + (bc*2 + dj) for di,dj in {0,1}
        # B block (br, bc) occupies V indices: (br*2 + di)*4 + (bc*2 + dj) for di,dj in {0,1}
        # C block (br, bc) occupies W indices: (bc*2 + dk)*4 + (br*2 + di) for di,dk in {0,1}
        #   Wait, W index is k*n + i where k is col of C, i is row of C
        #   C block (br, bc): rows br*2..br*2+1, cols bc*2..bc*2+1
        #   W index for C[i,k] = k*4 + i
        #   For block (br, bc): i = br*2 + di, k = bc*2 + dk
        #   W index = (bc*2 + dk)*4 + (br*2 + di)
        
        def a_block_vec(br, bc, coeff):
            """Return 16-vector for A block (br,bc) scaled by coeff."""
            vec = [0] * 16
            for di in range(2):
                for dj in range(2):
                    idx = (br * 2 + di) * 4 + (bc * 2 + dj)
                    vec[idx] = coeff
            return vec
        
        def b_block_vec(br, bc, coeff):
            """Return 16-vector for B block (br,bc) scaled by coeff."""
            vec = [0] * 16
            for di in range(2):
                for dj in range(2):
                    idx = (br * 2 + di) * 4 + (bc * 2 + dj)
                    vec[idx] = coeff
            return vec
        
        def c_block_vec(br, bc, coeff):
            """Return 16-vector for C block (br,bc) scaled by coeff."""
            vec = [0] * 16
            for di in range(2):
                for dk in range(2):
                    i = br * 2 + di
                    k = bc * 2 + dk
                    idx = k * 4 + i
                    vec[idx] = coeff
            return vec
        
        # Define the 7 terms
        # Each term: (A_terms, B_terms, C_terms)
        # A_terms: list of (block, coeff)
        # B_terms: list of (block, coeff)
        # C_terms: list of (block, coeff)
        
        terms = [
            # M1 = (A00 + A11)(B00 + B11)
            {
                'A': [((0,0), 1), ((1,1), 1)],
                'B': [((0,0), 1), ((1,1), 1)],
                'C': [((0,0), 1), ((1,1), 1)]
            },
            # M2 = (A10 + A11)B00
            {
                'A': [((1,0), 1), ((1,1), 1)],
                'B': [((0,0), 1)],
                'C': [((1,0), 1), ((1,1), -1)]
            },
            # M3 = A00(B01 - B11)
            {
                'A': [((0,0), 1)],
                'B': [((0,1), 1), ((1,1), -1)],
                'C': [((0,1), 1), ((1,1), -1)]
            },
            # M4 = A11(B10 - B00)
            {
                'A': [((1,1), 1)],
                'B': [((1,0), 1), ((0,0), -1)],
                'C': [((0,0), -1), ((1,0), 1)]
            },
            # M5 = (A00 + A01)B11
            {
                'A': [((0,0), 1), ((0,1), 1)],
                'B': [((1,1), 1)],
                'C': [((0,1), 1), ((1,1), -1)]
            },
            # M6 = (A10 - A00)(B00 + B01)
            {
                'A': [((1,0), 1), ((0,0), -1)],
                'B': [((0,0), 1), ((0,1), 1)],
                'C': [((0,0), 1)]
            },
            # M7 = (A01 - A11)(B10 + B11)
            {
                'A': [((0,1), 1), ((1,1), -1)],
                'B': [((1,0), 1), ((1,1), 1)],
                'C': [((0,0), -1)]
            },
        ]
        
        for t_idx, term in enumerate(terms):
            col = col_base + t_idx
            
            # Build U vector
            u_vec = [0] * 16
            for (br, bc), coeff in term['A']:
                block_vec = a_block_vec(br, bc, coeff)
                for idx in range(16):
                    u_vec[idx] += block_vec[idx]
            
            # Build V vector (with offset)
            v_vec = [0] * 16
            for (br, bc), coeff in term['B']:
                block_vec = b_block_vec(br, bc, coeff)
                for idx in range(16):
                    v_vec[idx] += block_vec[idx]
            
            # Build W vector (with offset)
            w_vec = [0] * 16
            for (br, bc), coeff in term['C']:
                block_vec = c_block_vec(br, bc, coeff)
                for idx in range(16):
                    w_vec[idx] += block_vec[idx]
            
            # Fill in U, V, W
            for idx in range(16):
                U[idx][col] += u_vec[idx]
                V[idx + v_offset * 4][col] += v_vec[idx]
                W[idx + w_offset * 4][col] += w_vec[idx]
    
    # First 4x4x4 block: v_offset=0, w_offset=0, col_base=0
    add_strassen_block(0, 0, 0)
    
    # Second 4x4x4 block: v_offset=1 (cols 4-7), w_offset=1 (cols 4-7), col_base=7
    add_strassen_block(1, 1, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
