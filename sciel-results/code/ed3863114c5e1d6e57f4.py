"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 7
    # We build a rank-49 decomposition using a Kronecker-product-like structure:
    # The 2x2x7 multiplication tensor is decomposed as the tensor product of
    # a 2x2x2 matrix multiplication (rank 7, Strassen) and a 2x1x7 vector-matrix 
    # multiplication (rank 7, schoolbook for single-column extraction).
    #
    # The idea: for each column l of B (l in 0..6), we treat the computation of
    # column l of C as a 2x2x1 matrix-vector product, which is a slice of the
    # 2x2x2 multiplication tensor. We use Strassen's 7 multiplications for the
    # 2x2x2 part, and for each Strassen term, we have 7 independent "column" 
    # decompositions (one per column of B), giving 7*7 = 49 total terms.
    #
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = A11 * (B12 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = (A11 + A12) * B22
    # M4 = A22 * (B11 + B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A11 - A21) * (B12 + B22)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # Reconstruction:
    # C11 = M5 + M4 - M3 - M7
    # C12 = M1 + M7
    # C21 = M2 + M3
    # C22 = M5 + M1 - M2 - M4
    #
    # For our 2x2x7 case, B is 2x7, so we have B[j, l] for j in {0,1}, l in 0..6.
    # We view each column l as a "2x1 matrix" B_l = [B[0,l], B[1,l]].
    # The Strassen decomposition applies to the 2x2x2 tensor where the "B" is 2x2.
    # But here, for each column l, we only have one column of B.
    #
    # Actually, let me think more carefully. The 2x2x2 multiplication tensor T_2x2x2
    # has indices (i,j) for A, (j,k) for B, (k,i) for C.
    # For 2x2x7, the tensor has indices (i,j) for A, (j,l) for B, (l,i) for C.
    # This is like 7 copies of the 2x2x1 tensor (matrix-vector product), sharing A.
    #
    # The rank of the 2x2x1 tensor (computing a 2x1 vector from 2x2 @ 2x1) is 2
    # (schoolbook). So naively 7*2 = 14, but A is shared.
    #
    # Let me use a different approach: the Kronecker product structure.
    #
    # Consider that the 2x2x7 tensor can be written as:
    # T = sum_{i,j,l} A[i,j] * B[j,l] * C[i,l]
    #
    # We can write this as a sum over 49 terms using the following construction:
    # For each Strassen term t (0..6) and each column l (0..6), define:
    # U[i,j] contribution from Strassen term t
    # V[j,l] contribution from column l
    # W[i,l] contribution from Strassen term t and column l
    #
    # Let me define the Strassen vectors for A, B, C in the 2x2x2 case:
    # A indices: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    #
    # Strassen terms (u, v, w) where u, v, w are 4-vectors:
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1 = A11 * (B12 + B22)
        ([1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]),
        # M2 = (A21 + A22) * B11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3 = (A11 + A12) * B22
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
        # M4 = A22 * (B11 + B21)
        ([0, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M5 = (A11 + A22) * (B11 + B22)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M6 = (A11 - A21) * (B12 + B22)
        ([1, 0, -1, 0], [0, 1, 0, 1], [0, 0, 0, 0]),
        # M7 = (A12 - A22) * (B21 + B22)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 14 x 49
    W = [[0] * rank for _ in range(p * n)]  # 14 x 49
    
    # For each Strassen term t and column l, we create a term.
    # The column index is col = t * p + l = t * 7 + l
    
    for t in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
        for l in range(p):
            col = t * p + l
            # U: A indices are (i, j) -> index i*m + j = i*2 + j
            # u_coeffs are for A[0,0], A[0,1], A[1,0], A[1,1]
            # So U[0][col] = u_coeffs[0]  (A[0,0])
            #    U[1][col] = u_coeffs[1]  (A[0,1])
            #    U[2][col] = u_coeffs[2]  (A[1,0])
            #    U[3][col] = u_coeffs[3]  (A[1,1])
            U[0][col] = u_coeffs[0]
            U[1][col] = u_coeffs[1]
            U[2][col] = u_coeffs[2]
            U[3][col] = u_coeffs[3]
            
            # V: B indices are (j, l) -> index j*p + l
            # For column l, we only have B[0, l] and B[1, l]
            # In the 2x2x2 Strassen, v_coeffs are for B[0,0], B[0,1], B[1,0], B[1,1]
            # For column l, B[0, l] corresponds to B[0,0] in the 2x2 view,
            # and B[1, l] corresponds to B[1,0] in the 2x2 view.
            # So V[0*p + l][col] = v_coeffs[0]  (B[0, l])
            #    V[1*p + l][col] = v_coeffs[2]  (B[1, l])
            V[0 * p + l][col] = v_coeffs[0]
            V[1 * p + l][col] = v_coeffs[2]
            
            # W: C indices are (l, i) -> index l*n + i = l*2 + i
            # w_coeffs are for C[0,0], C[0,1], C[1,0], C[1,1]
            # For column l, C[0, l] corresponds to C[0,0] in the 2x2 view,
            # and C[1, l] corresponds to C[1,0] in the 2x2 view.
            # So W[l*n + 0][col] = w_coeffs[0]  (C[0, l])
            #    W[l*n + 1][col] = w_coeffs[2]  (C[1, l])
            W[l * n + 0][col] = w_coeffs[0]
            W[l * n + 1][col] = w_coeffs[2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
