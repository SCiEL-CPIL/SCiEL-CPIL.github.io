import numpy as np

def construct():
    n = 5
    # Total rank 50
    rank = 50
    
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    # Helper to add a block decomposition
    # a_rows, a_cols: A block indices
    # b_rows, b_cols: B block indices
    # c_rows, c_cols: C block indices
    # inner_dim: size of the inner dimension (sum over k)
    # We use trivial decomposition for each block product
    # Rank contribution = inner_dim * c_cols
    
    def add_block(a_r, a_c, b_r, b_c, c_r, c_c, inner_dim, current_rank):
        # A is a_r x a_c, B is b_r x b_c, C is c_r x c_c
        # Actually, A block is (c_r, inner_dim), B block is (inner_dim, c_c)
        # So a_c should be inner_dim, b_r should be inner_dim
        # a_r = c_r, b_c = c_c
        
        block_rank = inner_dim * c_c
        
        for k in range(inner_dim):
            for j in range(c_c):
                idx = current_rank
                # U index: A[row, col] where row in a_r, col = k
                # Global A index: (a_r_start + row) * 5 + (a_c_start + k)
                # V index: B[row, col] where row = k, col in b_c
                # Global B index: (b_r_start + k) * 5 + (b_c_start + j)
                # W index: C[row, col] where row in c_r, col = j
                # Global C index: (c_c_start + j) * 5 + (c_r_start + row)
                
                for i in range(c_r):
                    u_idx = (a_r + i) * 5 + (a_c + k)
                    v_idx = (b_r + k) * 5 + (b_c + j)
                    w_idx = (c_c + j) * 5 + (c_r + i)
                    
                    U[u_idx, idx] = 1
                    V[v_idx, idx] = 1
                    W[w_idx, idx] = 1
                
                current_rank += 1
        
        return current_rank

    r = 0
    # C11 (3x3) = A11(3x3)B11(3x3) + A12(3x2)B21(2x3)
    # A11: rows 0-2, cols 0-2. B11: rows 0-2, cols 0-2. C11: rows 0-2, cols 0-2.
    # Inner dim 3. C cols 3.
    r = add_block(0, 0, 0, 0, 0, 0, 3, r) # A11 B11
    
    # A12: rows 0-2, cols 3-4. B21: rows 3-4, cols 0-2. C11: rows 0-2, cols 0-2.
    # A12 is 3x2. B21 is 2x3.
    # Inner dim 2. C cols 3.
    # Note: A12 cols are 3,4. B21 rows are 3,4.
    # In add_block, a_c is start col of A. b_r is start row of B.
    r = add_block(0, 3, 3, 0, 0, 0, 2, r) # A12 B21
    
    # C12 (3x2) = A11(3x3)B12(3x2) + A12(3x2)B22(2x2)
    # A11: rows 0-2, cols 0-2. B12: rows 0-2, cols 3-4. C12: rows 0-2, cols 3-4.
    # Inner dim 3. C cols 2.
    r = add_block(0, 0, 0, 3, 0, 3, 3, r) # A11 B12
    
    # A12: rows 0-2, cols 3-4. B22: rows 3-4, cols 3-4. C12: rows 0-2, cols 3-4.
    # Inner dim 2. C cols 2.
    r = add_block(0, 3, 3, 3, 0, 3, 2, r) # A12 B22
    
    # C21 (2x3) = A21(2x3)B11(3x3) + A22(2x2)B21(2x3)
    # A21: rows 3-4, cols 0-2. B11: rows 0-2, cols 0-2. C21: rows 3-4, cols 0-2.
    # Inner dim 3. C cols 3.
    r = add_block(3, 0, 0, 0, 3, 0, 3, r) # A21 B11
    
    # A22: rows 3-4, cols 3-4. B21: rows 3-4, cols 0-2. C21: rows 3-4, cols 0-2.
    # Inner dim 2. C cols 3.
    r = add_block(3, 3, 3, 0, 3, 0, 2, r) # A22 B21
    
    # C22 (2x2) = A21(2x3)B12(3x2) + A22(2x2)B22(2x2)
    # A21: rows 3-4, cols 0-2. B12: rows 0-2, cols 3-4. C22: rows 3-4, cols 3-4.
    # Inner dim 3. C cols 2.
    r = add_block(3, 0, 0, 3, 3, 3, 3, r) # A21 B12
    
    # A22: rows 3-4, cols 3-4. B22: rows 3-4, cols 3-4. C22: rows 3-4, cols 3-4.
    # Inner dim 2. C cols 2.
    r = add_block(3, 3, 3, 3, 3, 3, 2, r) # A22 B22
    
    return {'U': U, 'V': V, 'W': W}
