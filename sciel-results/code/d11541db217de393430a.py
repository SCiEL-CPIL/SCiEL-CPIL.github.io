def construct():
    n, m, p = 4, 5, 5
    
    U = {}
    V = {}
    W = {}
    
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        # Add a rank-1 term
        key = (u_idx, v_idx, w_idx)
        if key not in U:
            U[key] = 0
            V[key] = 0
            W[key] = 0
        U[key] += coeff_u
        V[key] += coeff_v
        W[key] += coeff_w
    
    # Actually, let me use a different approach. Each rank-1 term is a column in U, V, W.
    # Let me collect all terms as lists.
    
    terms = []  # Each term is (u_dict, v_dict, w_dict) where dicts map index->coeff
    
    def add_rank1(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts {index: coeff}"""
        terms.append((u_entries, v_entries, w_entries))
    
    # Strassen helper for 2x2 * 2x2 -> 2x2
    def strassen_222(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        a_rows, a_cols: row/col indices in A for the 2x2 block
        b_rows, b_cols: row/col indices in B for the 2x2 block  
        c_rows, c_cols: row/col indices in C for the 2x2 block
        """
        A00, A01 = a_rows[0]*m+a_cols[0], a_rows[0]*m+a_cols[1]
        A10, A11 = a_rows[1]*m+a_cols[0], a_rows[1]*m+a_cols[1]
        B00, B01 = b_rows[0]*p+b_cols[0], b_rows[0]*p+b_cols[1]
        B10, B11 = b_rows[1]*p+b_cols[0], b_rows[1]*p+b_cols[1]
        C00, C01 = c_rows[0]*n+c_cols[0], c_rows[0]*n+c_cols[1]
        C10, C11 = c_rows[1]*n+c_cols[0], c_rows[1]*n+c_cols[1]
        
        # M1 = (A00+A11)*(B00+B11)
        add_rank1({A00:1, A11:1}, {B00:1, B11:1}, {C00:1, C11:1})
        # M2 = (A10+A11)*B00
        add_rank1({A10:1, A11:1}, {B00:1}, {C10:1, C11:-1})
        # M3 = A00*(B01-B11)
        add_rank1({A00:1}, {B01:1, B11:-1}, {C01:1, C11:1})
        # M4 = A11*(B10-B00)
        add_rank1({A11:1}, {B10:1, B00:-1}, {C00:1, C10:1})
        # M5 = (A00+A01)*B11
        add_rank1({A00:1, A01:1}, {B11:1}, {C00:-1, C01:1})
        # M6 = (A10-A00)*(B00+B01)
        add_rank1({A10:1, A00:-1}, {B00:1, B01:1}, {C11:1})
        # M7 = (A01-A11)*(B10+B11)
        add_rank1({A01:1, A11:-1}, {B10:1, B11:1}, {C00:1, C11:1})
    
    # Schoolbook helper for (ra x ca) * (cb x cc) -> (rc x cc)
    # where ra, ca, cb, cc are sizes and we have index lists
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        for i_idx, i in enumerate(a_rows):
            for j_idx, j in enumerate(b_rows):
                # A[i,j] * B[j,k] for all k
                for k_idx, k in enumerate(b_cols):
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + c_cols[k_idx]
                    add_rank1({a_idx:1}, {b_idx:1}, {c_idx:1})
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (Strassen)
    strassen_222([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]  (schoolbook 2x2 * 2x3 -> 2x3)
    schoolbook_block([0,1], [0,1], [0,1], [2,3,4], [0,1], [2,3,4])
    
    # Block 3: A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2]  (schoolbook 2x3 * 3x2 -> 2x2)
    schoolbook_block([0,1], [2,3,4], [2,3,4], [0,1], [0,1], [0,1])
    
    # Block 4: A[0:2,2:5] * B[2:5,2:5] -> C[0:2,2:5]  (schoolbook 2x3 * 3x3 -> 2x3)
    schoolbook_block([0,1], [2,3,4], [2,3,4], [2,3,4], [0,1], [2,3,4])
    
    # Block 5: A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]  (Strassen)
    strassen_222([2,3], [0,1], [0,1], [0,1], [2,3], [0,1])
    
    # Block 6: A[2:4,0:2] * B[0:2,2:5] -> C[2:4,2:5]  (schoolbook 2x2 * 2x3 -> 2x3)
    schoolbook_block([2,3], [0,1], [0,1], [2,3,4], [2,3], [2,3,4])
    
    # Block 7: A[2:4,2:5] * B[2:5,0:2] -> C[2:4,0:2]  (schoolbook 2x3 * 3x2 -> 2x2)
    schoolbook_block([2,3], [2,3,4], [2,3,4], [0,1], [2,3], [0,1])
    
    # Block 8: A[2:4,2:5] * B[2:5,2:5] -> C[2:4,2:5]  (schoolbook 2x3 * 3x3 -> 2x3)
    schoolbook_block([2,3], [2,3,4], [2,3,4], [2,3,4], [2,3], [2,3,4])
    
    # Now build U, V, W matrices
    rank = len(terms)
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U_mat[idx][r] = coeff
        for idx, coeff in v_dict.items():
            V_mat[idx][r] = coeff
        for idx, coeff in w_dict.items():
            W_mat[idx][r] = coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
