"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 * 4x4 matrix multiplication
    using Strassen's algorithm on 2x2x2 sub-problems.
    
    The 2x4 * 4x4 multiplication can be decomposed into 4 independent 2x2 * 2x2 multiplications:
    - A_L * B_TL -> C_LL
    - A_L * B_TR -> C_LR  
    - A_R * B_BL -> C_RL
    - A_R * B_BR -> C_RR
    
    Each 2x2x2 sub-problem has rank 7 via Strassen's algorithm.
    Total rank = 4 * 7 = 28.
    
    Strassen's 7 multiplications for 2x2x2:
    M1 = (A11)(B11 + B22)
    M2 = (A11 + A21)(B22)
    M3 = (A22)(B11 + B12)
    M4 = (A22 - A11)(B22 - B21)
    M5 = (A11 + A12)(B22)
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 + A22)(B21 - B11)
    
    Output:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28  # 4 sub-problems * 7 multiplications each
    
    # U: shape (n*m, rank) = (8, 28)
    # V: shape (m*p, rank) = (16, 28)
    # W: shape (p*n, rank) = (8, 28), trace convention W[k*n + i, r]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 multiplications
    # Each term: (coeff_A, coeff_B, coeff_C) for each of the 7 multiplications
    # We need to map each 2x2 sub-problem to global indices
    
    # Define the 4 sub-problems:
    # Sub-problem s: A_rows, B_cols, C_rows, C_cols
    # s=0: A_L (rows 0,1), B_TL (cols 0,1), C_LL (rows 0,1, cols 0,1)
    # s=1: A_L (rows 0,1), B_TR (cols 2,3), C_LR (rows 0,1, cols 2,3)
    # s=2: A_R (rows 2,3), B_BL (cols 0,1), C_RL (rows 0,1, cols 0,1)
    # s=3: A_R (rows 2,3), B_BR (cols 2,3), C_RR (rows 0,1, cols 2,3)
    
    sub_problems = [
        # (A_row_offset, B_col_offset, C_row_offset, C_col_offset)
        (0, 0, 0, 0),  # s=0: A_L * B_TL -> C_LL
        (0, 2, 0, 2),  # s=1: A_L * B_TR -> C_LR
        (2, 0, 0, 0),  # s=2: A_R * B_BL -> C_RL
        (2, 2, 0, 2),  # s=3: A_R * B_BR -> C_RR
    ]
    
    # Strassen's 7 multiplications:
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A21) * B22
    # M3 = A22 * (B11 + B12)
    # M4 = (A22 - A11) * (B22 - B21)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 + A22) * (B21 - B11)
    
    # For each multiplication r (0-6), define the A and B coefficients
    # A coefficients: [coeff_A11, coeff_A12, coeff_A21, coeff_A22]
    # B coefficients: [coeff_B11, coeff_B12, coeff_B21, coeff_B22]
    
    strassen_A = [
        [1, 0, 0, 0],  # M1: A11
        [1, 0, 1, 0],  # M2: A11 + A21
        [0, 0, 0, 1],  # M3: A22
        [-1, 0, 0, 1], # M4: A22 - A11
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, 1],  # M7: A12 + A22
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [0, 0, 0, 1],  # M2: B22
        [1, 1, 0, 0],  # M3: B11 + B12
        [-1, 0, 1, 1], # M4: B22 - B21
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [-1, 0, 1, 0], # M7: B21 - B11
    ]
    
    # Output combination coefficients for C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_C = [
        [1, 0, 0, 1],  # C11: M1 + M4 - M5 + M7
        [0, 1, 1, 0],  # C12: M3 + M5
        [0, 1, 1, 0],  # C21: M2 + M4  (wait, let me recheck)
        [1, -1, 1, 0], # C22: M1 - M2 + M3 + M6
    ]
    
    # Let me re-derive Strassen's output combinations carefully:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each output position (c11, c12, c21, c22), the coefficients for (M1, M2, M3, M4, M5, M6, M7):
    strassen_C_full = [
        [1, 0, 0, 1, -1, 0, 1],   # C11 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],    # C12 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # C21 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0],   # C22 = M1 - M2 + M3 + M6
    ]
    
    # Now build the decomposition
    col = 0
    for s_idx, (a_row_off, b_col_off, c_row_off, c_col_off) in enumerate(sub_problems):
        for r in range(7):
            # A coefficients: which A elements contribute
            # In the sub-problem, A is 2x2 with local indices (a1, a2) where a1, a2 in {0,1}
            # Global A index: (a_row_off + a1, a2) -> row = a_row_off + a1, col = a2
            # U index: (a_row_off + a1) * m + a2
            
            for a1 in range(2):
                for a2 in range(2):
                    coeff = strassen_A[r][a1 * 2 + a2]
                    if coeff != 0:
                        u_idx = (a_row_off + a1) * m + a2
                        U[u_idx][col] += coeff
            
            # B coefficients: which B elements contribute
            # In the sub-problem, B is 2x2 with local indices (b1, b2) where b1, b2 in {0,1}
            # Global B index: (b1, b_col_off + b2) -> row = b1, col = b_col_off + b2
            # V index: b1 * p + (b_col_off + b2)
            
            for b1 in range(2):
                for b2 in range(2):
                    coeff = strassen_B[r][b1 * 2 + b2]
                    if coeff != 0:
                        v_idx = b1 * p + (b_col_off + b2)
                        V[v_idx][col] += coeff
            
            # W coefficients: which C elements are affected
            # In the sub-problem, C is 2x2 with local indices (c1, c2) where c1, c2 in {0,1}
            # Global C index: (c_row_off + c1, c_col_off + c2) -> row = c_row_off + c1, col = c_col_off + c2
            # W index: trace convention W[k*n + i] where k is column, i is row
            # So W index = (c_col_off + c2) * n + (c_row_off + c1)
            
            for c1 in range(2):
                for c2 in range(2):
                    coeff = strassen_C_full[c1 * 2 + c2][r]
                    if coeff != 0:
                        w_idx = (c_col_off + c2) * n + (c_row_off + c1)
                        W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
