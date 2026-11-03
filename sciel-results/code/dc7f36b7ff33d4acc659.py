"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    
    Uses a block-diagonal Strassen decomposition:
    - Split A (2x4) into A_L (2x2) and A_R (2x2)
    - Split B (4x4) into B_LL, B_RL, B_LR, B_RR (each 2x2)
    - C (2x4) = C_L (2x2) + C_R (2x2)
    
    Four independent 2x2x2 multiplications, each using Strassen (rank 7).
    Total rank = 4 * 7 = 28.
    
    Output factor follows the official trace convention W[k*n+i, r]
    where C = sum_r U[:,r] * V[:,r] * W[:,r] with appropriate reshaping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)  
    # W has shape (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2 matrix multiplication
    # Each term: (u_dict, v_dict, w_dict) where keys are linearized 2x2 indices
    # For 2x2 matrices: index = row*2 + col
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        # M2 = (A10+A11)(B00) -> 
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = (A00)(B11-B10) ->
        ({0: 1}, {1: 1, 2: -1}, {1: 1, 3: 1}),
        # M4 = (A11)(B10-B00) ->
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)(B11) ->
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01) ->
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) ->
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block structure:
    # Block 0: A_L (cols 0,1) * B_LL (rows 0,1; cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_RL (rows 2,3; cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_LR (rows 0,1; cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_RR (rows 2,3; cols 2,3) -> C_R (cols 2,3)
    
    # Global index mappings
    # A is 2x4, linearized as A[i*m + j] = A[i*4 + j]
    # B is 4x4, linearized as B[j*p + r] = B[j*4 + r]  
    # C is 2x4, but output uses trace convention W[k*n + i, r]
    # where C[i, r] = sum_k W[k*n + i, r] * ...
    # Actually, the convention is: C = sum_s U_s * V_s * W_s^T (appropriately)
    # Let me use the standard: C[i, j] = sum_{a,b} A[i,a] * B[a,j]
    # Tensor form: T[i, a, j] = sum_s U[i,a,s] * V[a,j,s] * W[j,i,s]
    # So W is indexed by (j, i) linearized as W[j*n + i, s]
    
    def global_A_idx(block, local_i, local_j):
        """Get global A index for block with local 2x2 indices."""
        i = local_i  # row in 0..1
        if block in [0, 2]:  # A_L: columns 0,1
            j = local_j
        else:  # A_R: columns 2,3
            j = 2 + local_j
        return i * m + j  # i*4 + j
    
    def global_B_idx(block, local_a, local_j):
        """Get global B index for block with local row (a) and col (j)."""
        if block in [0, 2]:  # B_LR or B_LL: rows 0,1
            a = local_a
        else:  # B_RR or B_RL: rows 2,3
            a = 2 + local_a
        if block in [0, 1]:  # B_LL or B_RL: cols 0,1
            j = local_j
        else:  # B_LR or B_RR: cols 2,3
            j = 2 + local_j
        return a * p + j  # a*4 + j
    
    def global_W_idx(block, local_j, local_i):
        """Get global W index using trace convention W[j*n + i, s]."""
        i = local_i  # row of C in 0..1
        if block in [0, 1]:  # C_L: cols 0,1
            j = local_j
        else:  # C_R: cols 2,3
            j = 2 + local_j
        return j * n + i  # j*2 + i
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = global_A_idx(b, li, lj)
                U[g_idx][col] += coeff
            
            for local_idx, coeff in v_coeffs.items():
                la = local_idx // 2
                lj = local_idx % 2
                g_idx = global_B_idx(b, la, lj)
                V[g_idx][col] += coeff
            
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2  # row i of local C
                lj = local_idx % 2   # col j of local C
                g_idx = global_W_idx(b, lj, li)
                W[g_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
