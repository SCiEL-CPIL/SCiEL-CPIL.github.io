"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # We will build the decomposition block by block.
    # Rank is estimated to be 50.
    # We'll use a list of tuples to store the terms first, then convert to matrices.
    # Term: (u_dict, v_dict, w_dict)
    terms = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict { (i, j): coeff }
        v_coeffs: dict { (j, k): coeff }
        w_coeffs: dict { (i, k): coeff } -> mapped to W index k*n+i
        """
        terms.append((u_coeffs, v_coeffs, w_coeffs))

    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds Strassen decomposition for a block.
        a_rows, a_cols: lists of global row/col indices for A block.
        b_rows, b_cols: lists of global row/col indices for B block.
        c_rows, c_cols: lists of global row/col indices for C block.
        Assumes 2x2 blocks.
        """
        # Local to global mapping
        # A local (0,0) -> global (a_rows[0], a_cols[0])
        # A local (0,1) -> global (a_rows[0], a_cols[1])
        # A local (1,0) -> global (a_rows[1], a_cols[0])
        # A local (1,1) -> global (a_rows[1], a_cols[1])
        
        # B local (0,0) -> global (b_rows[0], b_cols[0])
        # B local (0,1) -> global (b_rows[0], b_cols[1])
        # B local (1,0) -> global (b_rows[1], b_cols[0])
        # B local (1,1) -> global (b_rows[1], b_cols[1])
        
        # C local (0,0) -> global (c_rows[0], c_cols[0])
        # C local (0,1) -> global (c_rows[0], c_cols[1])
        # C local (1,0) -> global (c_rows[1], c_cols[0])
        # C local (1,1) -> global (c_rows[1], c_cols[1])
        
        # Strassen M1..M7
        # M1 = (A11+A22)(B11+B22)
        u1 = {(a_rows[0], a_cols[0]): 1, (a_rows[1], a_cols[1]): 1}
        v1 = {(b_rows[0], b_cols[0]): 1, (b_rows[1], b_cols[1]): 1}
        w1 = {(c_rows[0], c_cols[0]): 1, (c_rows[1], c_cols[1]): 1}
        # C11 += M1, C22 += M1
        # Wait, C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to accumulate W coefficients for each M.
        # It's easier to add 7 separate terms for M1..M7, and add their contributions to W.
        
        # M1
        w1 = {(c_rows[0], c_cols[0]): 1, (c_rows[1], c_cols[1]): 1}
        add_term(u1, v1, w1)
        
        # M2 = (A21+A22)B11
        u2 = {(a_rows[1], a_cols[0]): 1, (a_rows[1], a_cols[1]): 1}
        v2 = {(b_rows[0], b_cols[0]): 1}
        w2 = {(c_rows[1], c_cols[0]): 1, (c_rows[1], c_cols[1]): -1} # C21 += M2, C22 -= M2
        add_term(u2, v2, w2)
        
        # M3 = A11(B21-B22)
        u3 = {(a_rows[0], a_cols[0]): 1}
        v3 = {(b_rows[1], b_cols[0]): 1, (b_rows[1], b_cols[1]): -1}
        w3 = {(c_rows[0], c_cols[1]): 1, (c_rows[1], c_cols[1]): 1} # C12 += M3, C22 += M3
        add_term(u3, v3, w3)
        
        # M4 = A22(B12-B11)
        u4 = {(a_rows[1], a_cols[1]): 1}
        v4 = {(b_rows[0], b_cols[1]): 1, (b_rows[0], b_cols[0]): -1}
        w4 = {(c_rows[0], c_cols[0]): 1, (c_rows[1], c_cols[0]): 1} # C11 += M4, C21 += M4
        add_term(u4, v4, w4)
        
        # M5 = (A11+A12)B22
        u5 = {(a_rows[0], a_cols[0]): 1, (a_rows[0], a_cols[1]): 1}
        v5 = {(b_rows[1], b_cols[1]): 1}
        w5 = {(c_rows[0], c_cols[0]): -1, (c_rows[0], c_cols[1]): 1} # C11 -= M5, C12 += M5
        add_term(u5, v5, w5)
        
        # M6 = (A21-A11)(B11+B12)
        u6 = {(a_rows[1], a_cols[0]): 1, (a_rows[0], a_cols[0]): -1}
        v6 = {(b_rows[0], b_cols[0]): 1, (b_rows[0], b_cols[1]): 1}
        w6 = {(c_rows[1], c_cols[1]): 1} # C22 += M6
        add_term(u6, v6, w6)
        
        # M7 = (A12-A22)(B21+B22)
        u7 = {(a_rows[0], a_cols[1]): 1, (a_rows[1], a_cols[1]): -1}
        v7 = {(b_rows[1], b_cols[0]): 1, (b_rows[1], b_cols[1]): 1}
        w7 = {(c_rows[0], c_cols[0]): 1} # C11 += M7
        add_term(u7, v7, w7)

    def add_schoolbook(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds schoolbook decomposition for a block.
        """
        for i_idx, i in enumerate(a_rows):
            for k_idx, k in enumerate(c_cols):
                # Sum over j
                # A[i, j] * B[j, k]
                # In our decomposition, each (i, j, k) is a separate term?
                # Yes, schoolbook is sum of rank-1 tensors.
                for j_idx, j in enumerate(a_cols):
                    if j in b_rows: # Ensure j is valid for B
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(i, k): 1}
                        add_term(u, v, w)

    # Block 1: A[0:2, 0] * B[0, 0:9]
    # A rows 0,1. Col 0.
    # B row 0. Cols 0-8.
    # C rows 0,1. Cols 0-8.
    add_schoolbook([0, 1], [0], [0], list(range(9)), [0, 1], list(range(9)))
    
    # Block 2: A[0:2, 1:3] * B[1:3, 0:9]
    # A rows 0,1. Cols 1,2.
    # B rows 1,2. Cols 0-8.
    
    # Sub-block 2a: B cols 0,1
    add_strassen([0, 1], [1, 2], [1, 2], [0, 1], [0, 1], [0, 1])
    
    # Sub-block 2b: B cols 2,3
    add_strassen([0, 1], [1, 2], [1, 2], [2, 3], [0, 1], [2, 3])
    
    # Sub-block 2c: B cols 4,5
    add_strassen([0, 1], [1, 2], [1, 2], [4, 5], [0, 1], [4, 5])
    
    # Sub-block 2d: B cols 6,7
    add_strassen([0, 1], [1, 2], [1, 2], [6, 7], [0, 1], [6, 7])
    
    # Sub-block 2e: B col 8
    add_schoolbook([0, 1], [1, 2], [1, 2], [8], [0, 1], [8])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
        for (i, j), val in u_coeffs.items():
            U[i * m + j][r] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][r] = val
        for (i, k), val in w_coeffs.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
