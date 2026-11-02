"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the tensor decomposition of matrix multiplication.
    Dimensions: n x m * m x p = n x p
    n=2, m=4, p=8
    Uses a block decomposition with Strassen for 2x2x2 sub-problems.
    Rank: 56
    """
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # Coefficients for U (A part) for each M
    U_coeffs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # Coefficients for V (B part) for each M
    V_coeffs = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # Coefficients for W (C part) for each M
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Define the 8 sub-problems
    # (A_row_start, A_col_start, B_row_start, B_col_start)
    # A is 2x4, B is 4x8
    # A_L: rows 0-1, cols 0-1
    # A_R: rows 0-1, cols 2-3
    # B_T: rows 0-1, cols 0-7 (split into 4 blocks of width 2)
    # B_B: rows 2-3, cols 0-7 (split into 4 blocks of width 2)
    
    sub_problems = [
        # A_L, B_T blocks
        (0, 0, 0, 0), # A_L, B_T0
        (0, 0, 0, 2), # A_L, B_T1
        (0, 0, 0, 4), # A_L, B_T2
        (0, 0, 0, 6), # A_L, B_T3
        # A_R, B_B blocks
        (0, 2, 2, 0), # A_R, B_B0
        (0, 2, 2, 2), # A_R, B_B1
        (0, 2, 2, 4), # A_R, B_B2
        (0, 2, 2, 6), # A_R, B_B3
    ]
    
    for sp in sub_problems:
        a_r0, a_c0, b_r0, b_c0 = sp
        
        # Global indices for A elements (2x2 block)
        # A is indexed i*m + j
        a_idx_00 = a_r0 * m + a_c0
        a_idx_01 = a_r0 * m + a_c0 + 1
        a_idx_10 = (a_r0 + 1) * m + a_c0
        a_idx_11 = (a_r0 + 1) * m + a_c0 + 1
        a_indices = [a_idx_00, a_idx_01, a_idx_10, a_idx_11]
        
        # Global indices for B elements (2x2 block)
        # B is indexed j*p + k
        b_idx_00 = b_r0 * p + b_c0
        b_idx_01 = b_r0 * p + b_c0 + 1
        b_idx_10 = (b_r0 + 1) * p + b_c0
        b_idx_11 = (b_r0 + 1) * p + b_c0 + 1
        b_indices = [b_idx_00, b_idx_01, b_idx_10, b_idx_11]
        
        # Global indices for C elements (2x2 block)
        # C is indexed k*n + i
        # The C block corresponds to rows a_r0..a_r0+1 of C and cols b_c0..b_c0+1
        # Wait, C is n x p.
        # The sub-problem calculates a 2x2 block of C.
        # The rows of C are i=0,1. The cols of C are k=b_c0, b_c0+1.
        # So C elements are C[i, k].
        # Index in W is k*n + i.
        
        c_idx_00 = b_c0 * n + a_r0
        c_idx_01 = (b_c0 + 1) * n + a_r0
        c_idx_10 = b_c0 * n + (a_r0 + 1)
        c_idx_11 = (b_c0 + 1) * n + (a_r0 + 1)
        c_indices = [c_idx_00, c_idx_01, c_idx_10, c_idx_11]
        
        # Assign coefficients for the 7 multiplications
        for k in range(7):
            col = col_offset + k
            
            # Fill U
            for i in range(4):
                U[a_indices[i]][col] = U_coeffs[k][i]
                
            # Fill V
            for j in range(4):
                V[b_indices[j]][col] = V_coeffs[k][j]
                
            # Fill W
            for i in range(4):
                W[c_indices[i]][col] = W_coeffs[k][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
