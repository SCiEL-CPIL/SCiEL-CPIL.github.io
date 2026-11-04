"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build U, V, W as lists of lists (row-major)
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx):
        """Add a rank-1 term with coefficient 1."""
        U.append([1 if i == u_idx else 0 for i in range(n*m)])
        V.append([1 if i == v_idx else 0 for i in range(m*p)])
        W.append([1 if i == w_idx else 0 for i in range(p*n)])
    
    def add_scaled_term(u_idx, v_idx, w_idx, coeff):
        """Add a rank-1 term with coefficient."""
        if coeff == 0:
            return
        U.append([coeff if i == u_idx else 0 for i in range(n*m)])
        V.append([1 if i == v_idx else 0 for i in range(m*p)])
        W.append([1 if i == w_idx else 0 for i in range(p*n)])
    
    # Strassen's rank-7 algorithm for 2x2x2
    # Multiplies A (2x2) * B (2x2) -> C (2x2)
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global indices: A[i,j], B[j,k], C[i,k]
    def strassen_2x2x2(row_offset, col_offset, k_offset):
        """
        Add Strassen rank-7 terms for the 2x2x2 subproblem.
        row_offset: offset for rows of A and C
        col_offset: offset for columns of C
        k_offset: offset for columns of B (and rows of A's second dimension)
        
        A is (row_offset:row_offset+2, 0:2)
        B is (0:2, k_offset:k_offset+2)
        C is (row_offset:row_offset+2, col_offset:col_offset+2)
        """
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[row_offset, 0], A11 = A[row_offset+1, 1]
        # B00 = B[0, k_offset], B11 = B[1, k_offset+1]
        # C00 += M1, C11 += M1
        # U indices: A[row_offset, 0] and A[row_offset+1, 1]
        # V indices: B[0, k_offset] and B[1, k_offset+1]
        # W indices: C[row_offset, col_offset] and C[row_offset+1, col_offset+1]
        
        # We need to add terms that compute (A00+A11)(B00+B11)
        # This is 4 terms: A00*B00, A00*B11, A11*B00, A11*B11
        # But Strassen combines them into 7 multiplications total for the whole 2x2x2.
        
        # Let me implement the standard Strassen 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B10
        # M6 = (A01-A00)(B00+B10)
        # M7 = (A11-A10)(B01+B11)
        
        # C00 = M1 - M5 + M7
        # C01 = M3 + M6
        # C10 = M2 + M4
        # C11 = M1 - M2 + M4
        
        # Each Mi is a product of linear combinations of A and B entries.
        # For tensor decomposition, each Mi corresponds to a rank-1 term
        # where U has the A coefficients, V has the B coefficients, and W distributes to C.
        
        # Let me define helper to add a term with multiple U and V entries
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            """
            u_coeffs: dict mapping A index to coefficient
            v_coeffs: dict mapping B index to coefficient  
            w_coeffs: dict mapping C index to coefficient
            """
            # This creates a single rank-1 term
            # U vector has non-zero entries at A indices
            # V vector has non-zero entries at B indices
            # W vector has non-zero entries at C indices
            
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            for (i, j), coeff in u_coeffs.items():
                u_vec[i*m + j] = coeff
            for (j, k), coeff in v_coeffs.items():
                v_vec[j*p + k] = coeff
            for (i, k), coeff in w_coeffs.items():
                w_vec[k*n + i] = coeff
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
        
        # A indices: (row_offset+di, dj) for di in {0,1}, dj in {0,1}
        # B indices: (dj, k_offset+dk) for dj in {0,1}, dk in {0,1}
        # C indices: (row_offset+di, col_offset+dk) for di in {0,1}, dk in {0,1}
        
        A00 = (row_offset, 0)
        A01 = (row_offset, 1)
        A10 = (row_offset+1, 0)
        A11 = (row_offset+1, 1)
        
        B00 = (0, k_offset)
        B01 = (0, k_offset+1)
        B10 = (1, k_offset)
        B11 = (1, k_offset+1)
        
        C00 = (row_offset, col_offset)
        C01 = (row_offset, col_offset+1)
        C10 = (row_offset+1, col_offset)
        C11 = (row_offset+1, col_offset+1)
        
        # M1 = (A00+A11)(B00+B11)
        add_strassen_term(
            {A00: 1, A11: 1},
            {B00: 1, B11: 1},
            {C00: 1, C11: 1}
        )
        
        # M2 = (A10+A11)B00
        add_strassen_term(
            {A10: 1, A11: 1},
            {B00: 1},
            {C10: 1, C11: 1}
        )
        
        # M3 = A00(B01-B11)
        add_strassen_term(
            {A00: 1},
            {B01: 1, B11: -1},
            {C01: 1}
        )
        
        # M4 = A11(B10-B00)
        add_strassen_term(
            {A11: 1},
            {B10: 1, B00: -1},
            {C10: 1, C11: 1}
        )
        
        # M5 = (A00+A10)B10
        add_strassen_term(
            {A00: 1, A10: 1},
            {B10: 1},
            {C00: -1, C10: 1}
        )
        
        # M6 = (A01-A00)(B00+B10)
        add_strassen_term(
            {A01: 1, A00: -1},
            {B00: 1, B10: 1},
            {C01: 1, C11: -1}
        )
        
        # M7 = (A11-A10)(B01+B11)
        add_strassen_term(
            {A11: 1, A10: -1},
            {B01: 1, B11: 1},
            {C00: 1, C01: 1}
        )
    
    # Process 4 blocks of 2 columns each
    for block in range(4):
        base_k = block * 2
        
        # Strassen for 2x2x2: A[0:2, 0:2] * B[0:2, base_k:base_k+2] -> C[0:2, base_k:base_k+2]
        strassen_2x2x2(0, base_k, base_k)
        
        # Now handle the remaining terms involving index 2
        # C[0, base_k] and C[0, base_k+1] need A[0,2]*B[2, base_k] and A[0,2]*B[2, base_k+1]
        # C[1, base_k] and C[1, base_k+1] need A[1,2]*B[2, base_k] and A[1,2]*B[2, base_k+1]
        # C[2, base_k] and C[2, base_k+1] need A[2,0]*B[0, base_k] + A[2,1]*B[1, base_k] + A[2,2]*B[2, base_k]
        #                                  and A[2,0]*B[0, base_k+1] + A[2,1]*B[1, base_k+1] + A[2,2]*B[2, base_k+1]
        
        # Add schoolbook terms for these
        # A[0,2]*B[2, base_k] -> C[0, base_k]
        add_term(0*m+2, 2*p+base_k, base_k*n+0)
        # A[0,2]*B[2, base_k+1] -> C[0, base_k+1]
        add_term(0*m+2, 2*p+base_k+1, (base_k+1)*n+0)
        # A[1,2]*B[2, base_k] -> C[1, base_k]
        add_term(1*m+2, 2*p+base_k, base_k*n+1)
        # A[1,2]*B[2, base_k+1] -> C[1, base_k+1]
        add_term(1*m+2, 2*p+base_k+1, (base_k+1)*n+1)
        # A[2,0]*B[0, base_k] -> C[2, base_k]
        add_term(2*m+0, 0*p+base_k, base_k*n+2)
        # A[2,1]*B[1, base_k] -> C[2, base_k]
        add_term(2*m+1, 1*p+base_k, base_k*n+2)
        # A[2,2]*B[2, base_k] -> C[2, base_k]
        add_term(2*m+2, 2*p+base_k, base_k*n+2)
        # A[2,0]*B[0, base_k+1] -> C[2, base_k+1]
        add_term(2*m+0, 0*p+base_k+1, (base_k+1)*n+2)
        # A[2,1]*B[1, base_k+1] -> C[2, base_k+1]
        add_term(2*m+1, 1*p+base_k+1, (base_k+1)*n+2)
        # A[2,2]*B[2, base_k+1] -> C[2, base_k+1]
        add_term(2*m+2, 2*p+base_k+1, (base_k+1)*n+2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
