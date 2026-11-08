"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We'll build U, V, W as lists of lists (or use sparse representation)
    # Start with empty
    U_terms = []  # Each term is a list of (row, value) for the rank columns
    V_terms = []
    W_terms = []
    
    # Helper to add a scalar multiple of a rank-1 term
    def add_term(u_row, v_row, w_row, coeff=1):
        """Add a rank-1 term with given row indices and coefficient"""
        U_terms.append((u_row, coeff))
        V_terms.append((v_row, coeff))
        W_terms.append((w_row, coeff))
    
    # Helper to add a block of terms
    # For multiplying A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
    # where indices are (start, end) for rows/cols
    def add_block_decomposition(i1, i2, j1, j2, k1, k2, decomp_fn):
        """Add terms from a sub-decomposition"""
        decomp_fn(i1, i2, j1, j2, k1, k2)
    
    # Strassen 2x2x2 decomposition
    # Multiplies A[2x2] * B[2x2] -> C[2x2]
    # A indices: rows i1:i1+2, cols j1:j1+2
    # B indices: rows j1:j1+2, cols k1:k1+2
    # C indices: rows i1:i1+2, cols k1:k1+2
    def strassen_2x2(i1, i2, j1, j2, k1, k2):
        # A sub-blocks: A11=[i1,i1+1]x[j1,j1+1], A12=[i1,i1+1]x[j1+1,j1+2]
        #               A21=[i1+1,i1+2]x[j1,j1+1], A22=[i1+1,i1+2]x[j1+1,j1+2]
        # B sub-blocks: B11=[j1,j1+1]x[k1,k1+1], B12=[j1,j1+1]x[k1+1,k1+2]
        #               B21=[j1+1,j1+2]x[k1,k1+1], B22=[j1+1,j1+2]x[k1+1,k1+2]
        
        # M1 = (A11 + A22)(B11 + B22)
        # A part: rows i1:i1+1 cols j1:j1+1 (coeff 1) + rows i1+1:i1+2 cols j1+1:j1+2 (coeff 1)
        # B part: rows j1:j1+1 cols k1:k1+1 (coeff 1) + rows j1+1:j1+2 cols k1+1:k1+2 (coeff 1)
        # C part: rows i1:i1+1 cols k1:k1+1 (coeff 1)
        add_term((i1, 1, j1, 1), (j1, 1, k1, 1), (i1, 1, k1, 1), 1)
        add_term((i1, 1, j1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1, 1), 1)
        add_term((i1+1, 1, j1+1, 1), (j1, 1, k1, 1), (i1, 1, k1, 1), 1)
        add_term((i1+1, 1, j1+1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1, 1), 1)
        
        # M2 = (A21 + A22) * B11
        # A part: rows i1+1:i1+2 cols j1:j1+1 (coeff 1) + rows i1+1:i1+2 cols j1+1:j1+2 (coeff 1)
        # B part: rows j1:j1+1 cols k1:k1+1 (coeff 1)
        # C part: rows i1+1:i1+2 cols k1:k1+1 (coeff 1)
        add_term((i1+1, 1, j1, 1), (j1, 1, k1, 1), (i1+1, 1, k1, 1), 1)
        add_term((i1+1, 1, j1+1, 1), (j1, 1, k1, 1), (i1+1, 1, k1, 1), 1)
        
        # M3 = A11 * (B12 - B22)
        # A part: rows i1:i1+1 cols j1:j1+1 (coeff 1)
        # B part: rows j1:j1+1 cols k1+1:k1+2 (coeff 1) - rows j1+1:j1+2 cols k1+1:k1+2 (coeff 1)
        # C part: rows i1:i1+1 cols k1+1:k1+2 (coeff 1)
        add_term((i1, 1, j1, 1), (j1, 1, k1+1, 1), (i1, 1, k1+1, 1), 1)
        add_term((i1, 1, j1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1+1, 1), -1)
        
        # M4 = A22 * (B21 - B11)
        # A part: rows i1+1:i1+2 cols j1+1:j1+2 (coeff 1)
        # B part: rows j1+1:j1+2 cols k1:k1+1 (coeff 1) - rows j1:j1+1 cols k1:k1+1 (coeff 1)
        # C part: rows i1+1:i1+2 cols k1:k1+1 (coeff 1)
        add_term((i1+1, 1, j1+1, 1), (j1+1, 1, k1, 1), (i1+1, 1, k1, 1), 1)
        add_term((i1+1, 1, j1+1, 1), (j1, 1, k1, 1), (i1+1, 1, k1, 1), -1)
        
        # M5 = (A11 + A12) * B22
        # A part: rows i1:i1+1 cols j1:j1+1 (coeff 1) + rows i1:i1+1 cols j1+1:j1+2 (coeff 1)
        # B part: rows j1+1:j1+2 cols k1+1:k1+2 (coeff 1)
        # C part: rows i1:i1+1 cols k1+1:k1+2 (coeff 1)
        add_term((i1, 1, j1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1+1, 1), 1)
        add_term((i1, 1, j1+1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1+1, 1), 1)
        
        # M6 = (A21 - A11) * (B11 + B12)
        # A part: rows i1+1:i1+2 cols j1:j1+1 (coeff 1) - rows i1:i1+1 cols j1:j1+1 (coeff 1)
        # B part: rows j1:j1+1 cols k1:k1+1 (coeff 1) + rows j1:j1+1 cols k1+1:k1+2 (coeff 1)
        # C part: rows i1+1:i1+2 cols k1+1:k1+2 (coeff 1)
        add_term((i1+1, 1, j1, 1), (j1, 1, k1, 1), (i1+1, 1, k1+1, 1), 1)
        add_term((i1+1, 1, j1, 1), (j1, 1, k1+1, 1), (i1+1, 1, k1+1, 1), 1)
        add_term((i1, 1, j1, 1), (j1, 1, k1, 1), (i1+1, 1, k1+1, 1), -1)
        add_term((i1, 1, j1, 1), (j1, 1, k1+1, 1), (i1+1, 1, k1+1, 1), -1)
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A part: rows i1:i1+1 cols j1+1:j1+2 (coeff 1) - rows i1+1:i1+2 cols j1+1:j1+2 (coeff 1)
        # B part: rows j1+1:j1+2 cols k1:k1+1 (coeff 1) + rows j1+1:j1+2 cols k1+1:k1+2 (coeff 1)
        # C part: rows i1:i1+1 cols k1+1:k1+2 (coeff 1)
        add_term((i1, 1, j1+1, 1), (j1+1, 1, k1, 1), (i1, 1, k1+1, 1), 1)
        add_term((i1, 1, j1+1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1+1, 1), 1)
        add_term((i1+1, 1, j1+1, 1), (j1+1, 1, k1, 1), (i1, 1, k1+1, 1), -1)
        add_term((i1+1, 1, j1+1, 1), (j1+1, 1, k1+1, 1), (i1, 1, k1+1, 1), -1)
    
    # Wait, the above approach is wrong. Let me think more carefully.
    # 
    # The tensor decomposition works as: C = sum_t U[:,t] * V[:,t] * W[:,t]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the product is: C[i,k] = sum_t sum_j U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # So each rank-1 term is defined by:
    # - U has 1's at positions corresponding to (i, j) pairs
    # - V has 1's at positions corresponding to (j, k) pairs  
    # - W has 1's at positions corresponding to (k, i) pairs
    # - With appropriate coefficients
    #
    # For Strassen, the 7 products are:
    # M1 = (A11+A22)(B11+B22)
    #   A terms: (i1,j1), (i1,j1+1), (i1+1,j1), (i1+1,j1+1) -- wait no
    #   A11 = rows i1:i1+1, cols j1:j1+1 -- so the (i,j) pairs are (i1,j1)
    #   A22 = rows i1+1:i1+2, cols j1+1:j1+2 -- so the (i,j) pairs are (i1+1,j1+1)
    #   So U has entries at (i1,j1) with coeff 1, (i1+1,j1+1) with coeff 1
    #   B11 = rows j1:j1+1, cols k1:k1+1 -- (j,k) pairs: (j1,k1)
    #   B22 = rows j1+1:j1+2, cols k1+1:k1+2 -- (j,k) pairs: (j1+1,k1+1)
    #   So V has entries at (j1,k1) with coeff 1, (j1+1,k1+1) with coeff 1
    #   C11 = rows i1:i1+1, cols k1:k1+1 -- (k,i) pairs: (k1,i1)
    #   So W has entry at (k1,i1) with coeff 1
    #
    # But this is getting complicated. Let me use a different approach.
    # 
    # I'll directly construct the U, V, W matrices by building up the contributions.
    # 
    # Let me restart with a cleaner approach. I'll build U, V, W as dictionaries
    # mapping (row, col) -> value, where col is the rank index.
    
    U_dict = {}  # (row, rank_col) -> value
    V_dict = {}
    W_dict = {}
    next_rank = 0
    
    def add_rank_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term where u_entries, v_entries, w_entries are lists of (row, coeff)"""
        nonlocal next_rank
        for row, coeff in u_entries:
            key = (row, next_rank)
            U_dict[key] = U_dict.get(key, 0) + coeff
        for row, coeff in v_entries:
            key = (row, next_rank)
            V_dict[key] = V_dict.get(key, 0) + coeff
        for row, coeff in w_entries:
            key = (row, next_rank)
            W_dict[key] = W_dict.get(key, 0) + coeff
        next_rank += 1
    
    # Strassen 2x2x2: multiplies A[2x2] * B[2x2] -> C[2x2]
    # A rows: i1, i1+1; A cols: j1, j1+1
    # B rows: j1, j1+1; B cols: k1, k1+1
    # C rows: i1, i1+1; C cols: k1, k1+1
    def strassen_2x2(i1, j1, k1):
        # M1 = (A11 + A22)(B11 + B22)
        # A11: (i1,j1), A22: (i1+1,j1+1)
        # B11: (j1,k1), B22: (j1+1,k1+1)
        # C11: (k1,i1)
        add_rank_term(
            [(i1*m+j1, 1), (i1+1)*m+j1+1, 1],  # This is wrong syntax
            [], []
        )
        # Let me fix the syntax
    
    # Let me rewrite more carefully
    U_dict = {}
    V_dict = {}
    W_dict = {}
    next_rank = 0
    
    def add_rank_term(u_entries, v_entries, w_entries):
        nonlocal next_rank
        for row, coeff in u_entries:
            key = (row, next_rank)
            U_dict[key] = U_dict.get(key, 0) + coeff
        for row, coeff in v_entries:
            key = (row, next_rank)
            V_dict[key] = V_dict.get(key, 0) + coeff
        for row, coeff in w_entries:
            key = (row, next_rank)
            W_dict[key] = W_dict.get(key, 0) + coeff
        next_rank += 1
    
    def strassen_2x2(i1, j1, k1):
        # M1 = (A11 + A22)(B11 + B22)
        add_rank_term(
            [(i1*m+j1, 1), ((i1+1)*m+(j1+1), 1)],
            [(j1*p+k1, 1), ((j1+1)*p+(k1+1), 1)],
            [(k1*n+i1, 1)]
        )
        # M2 = (A21 + A22) * B11
        add_rank_term(
            [((i1+1)*m+j1, 1), ((i1+1)*m+(j1+1), 1)],
            [(j1*p+k1, 1)],
            [(k1*n+(i1+1), 1)]
        )
        # M3 = A11 * (B12 - B22)
        add_rank_term(
            [(i1*m+j1, 1)],
            [(j1*p+(k1+1), 1), ((j1+1)*p+(k1+1), -1)],
            [(k1+1)*n+i1, 1]
        )
        # M4 = A22 * (B21 - B11)
        add_rank_term(
            [((i1+1)*m+(j1+1), 1)],
            [((j1+1)*p+k1, 1), (j1*p+k1, -1)],
            [(k1*n+(i1+1), 1)]
        )
        # M5 = (A11 + A12) * B22
        add_rank_term(
            [(i1*m+j1, 1), (i1*m+(j1+1), 1)],
            [((j1+1)*p+(k1+1), 1)],
            [(k1+1)*n+i1, 1]
        )
        # M6 = (A21 - A11) * (B11 + B12)
        add_rank_term(
            [((i1+1)*m+j1, 1), (i1*m+j1, -1)],
            [(j1*p+k1, 1), (j1*p+(k1+1), 1)],
            [(k1+1)*n+(i1+1), 1]
        )
        # M7 = (A12 - A22) * (B21 + B22)
        add_rank_term(
            [(i1*m+(j1+1), 1), ((i1+1)*m+(j1+1), -1)],
            [((j1+1)*p+k1, 1), ((j1+1)*p+(k1+1), 1)],
            [(k1+1)*n+i1, 1]
        )
    
    # Schoolbook for a block: A[a1:a2, b1:b2] * B[b1:b2, c1:c2] -> C[a1:a2, c1:c2]
    def schoolbook_block(i1, i2, j1, j2, k1, k2):
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    add_rank_term(
                        [(i*m+j, 1)],
                        [(j*p+k, 1)],
                        [(k*n+i, 1)]
                    )
    
    # Matrix-vector: A[2x3] * B[3x1] -> C[2x1]
    # This is just 2*3*1 = 6 terms, same as schoolbook
    # But we can potentially do better? No, 2x3x1 has rank 6 (it's a product of two matrices)
    # Actually, for a fixed k, A * B[:,k] is a 2x3 times 3x1, which has rank at most 2
    # But the tensor is over all i,j,k, so it's still 6 terms minimum
    # Let me just use schoolbook for these
    
    # Now build the full 5x5x5 decomposition
    # Split: rows 0-1 (2 rows), rows 2-4 (3 rows)
    # cols same
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] -- Strassen
    strassen_2x2(0, 0, 0)
    
    # Block (0,1): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # A is 2x2, B is 2x3, C is 2x3
    # For each k in 2,3,4: A * B[:,k] is 2x3 times 3x1 -- wait, B is 2x3, so B[:,k] doesn't make sense
    # B has rows j (0,1) and cols k (2,3,4). So for fixed k, B[:,k] is a 2x1 vector
    # A * B[:,k] is 2x2 times 2x1 = 2x1, rank at most 2
    # Total: 3 * 2 = 6 terms? No, schoolbook is 2*2*3 = 12
    # Actually the tensor is T[i,j,k] = A[i,j] * B[j,k]
    # For i in 0,1; j in 0,1; k in 2,3,4: that's 2*2*3 = 12 terms
    # Can we do better? The matrix A is 2x2, B is 2x3. The product is 2x3.
    # The rank of the tensor is the minimum number of rank-1 tensors
    # For a fixed k, the slice is A * B[:,k] which is a 2x2 matrix times a 2x1 vector
    # This has rank at most 2 (as a matrix), so the tensor slice has rank at most 2
    # But across different k, we might be able to share
    # Let's just use schoolbook for now: 12 terms
    schoolbook_block(0, 2, 0, 2, 2, 5)
    
    # Block (0,2): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # A is 2x3, B is 3x2, C is 2x2
    # Schoolbook: 2*3*2 = 12
    schoolbook_block(0, 2, 2, 5, 0, 2)
    
    # Block (0,3): A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # A is 2x3, B is 3x3, C is 2x3
    # Schoolbook: 2*3*3 = 18
    schoolbook_block(0, 2, 2, 5, 2, 5)
    
    # Block (1,0): A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # A is 3x2, B is 2x2, C is 3x2
    # Schoolbook: 3*2*2 = 12
    schoolbook_block(2, 5, 0, 2, 0, 2)
    
    # Block (1,1): A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # A is 3x2, B is 2x3, C is 3x3
    # Schoolbook: 3*2*3 = 18
    schoolbook_block(2, 5, 0, 2, 2, 5)
    
    # Block (1,2): A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # A is 3x3, B is 3x2, C is 3x2
    # Schoolbook: 3*3*2 = 18
    schoolbook_block(2, 5, 2, 5, 0, 2)
    
    # Block (1,3): A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # A is 3x3, B is 3x3, C is 3x3
    # Schoolbook: 3*3*3 = 27
    schoolbook_block(2, 5, 2, 5, 2, 5)
    
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    # Hmm, that's 124, not 120. Let me recount.
    # Strassen: 7
    # (0,1): 12
    # (0,2): 12
    # (0,3): 18
    # (1,0): 12
    # (1,1): 18
    # (1,2): 18
    # (1,3): 27
    # Total: 7+12+12+18+12+18+18+27 = 124
    
    # To get 120, I need to save 4 more. 
    # Let me reconsider. Maybe I can use a better decomposition for some blocks.
    # 
    # For block (0,1): A[2x2] * B[2x3] -> C[2x3]
    # Can I use Strassen-like tricks? 
    # Actually, let me just go with 124 for now and see if it's valid.
    # 125/124 = 1.008, which is better than 1.0
    
    # Convert dictionaries to lists
    rank = next_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, col), val in U_dict.items():
        U[row][col] = val
    for (row, col), val in V_dict.items():
        V[row][col] = val
    for (row, col), val in W_dict.items():
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

