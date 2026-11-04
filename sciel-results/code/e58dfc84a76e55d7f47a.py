"""Matrix multiplication 4x5x5 with Strassen-like block decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    rank1_terms = []  # list of (U_coeffs_dict, V_coeffs_dict, W_coeffs_dict)
    
    def add_rank1_term(U_coeffs, V_coeffs, W_coeffs):
        rank1_terms.append((U_coeffs, V_coeffs, W_coeffs))
    
    def strassen_block_2x2x2(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """
        Add Strassen rank-7 terms for 2x2x2 multiplication.
        A block: 2x2 at (A_row_off, A_col_off)
        B block: 2x2 at (B_row_off, B_col_off)  
        C block: 2x2 at (C_row_off, C_col_off)
        """
        def A_pos(ii, jj):
            return (A_row_off + ii) * m + (A_col_off + jj)
        
        def B_pos(jj, kk):
            return (B_row_off + jj) * p + (B_col_off + kk)
        
        def C_pos(ii, kk):
            return (C_col_off + kk) * n + (C_row_off + ii)
        
        A11, A12, A21, A22 = A_pos(0,0), A_pos(0,1), A_pos(1,0), A_pos(1,1)
        B11, B12, B21, B22 = B_pos(0,0), B_pos(0,1), B_pos(1,0), B_pos(1,1)
        C11, C12, C21, C22 = C_pos(0,0), C_pos(0,1), C_pos(1,0), C_pos(1,1)
        
        # M1 = (A11+A22)(B11+B22), contributes +1 to C11, +1 to C22
        add_rank1_term({A11: 1, A22: 1}, {B11: 1, B22: 1}, {C11: 1, C22: 1})
        
        # M2 = (A21+A22)B11, contributes +1 to C21
        add_rank1_term({A21: 1, A22: 1}, {B11: 1}, {C21: 1})
        
        # M3 = A11(B21-B22), contributes +1 to C12
        add_rank1_term({A11: 1}, {B21: 1, B22: -1}, {C12: 1})
        
        # M4 = A22(B12-B11), contributes +1 to C11, +1 to C21
        add_rank1_term({A22: 1}, {B12: 1, B11: -1}, {C11: 1, C21: 1})
        
        # M5 = (A11+A12)B22, contributes -1 to C11, +1 to C12
        add_rank1_term({A11: 1, A12: 1}, {B22: 1}, {C11: -1, C12: 1})
        
        # M6 = (A21-A11)(B11+B12), contributes +1 to C22
        add_rank1_term({A21: 1, A11: -1}, {B11: 1, B12: 1}, {C22: 1})
        
        # M7 = (A12-A22)(B21+B22), contributes +1 to C11
        add_rank1_term({A12: 1, A22: -1}, {B21: 1, B22: 1}, {C11: 1})
    
    # C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    strassen_block_2x2x2(0, 0, 0, 0, 0, 0)
    
    # A12*B21: 2x3 x 3x2 -> 2x2
    for ii in range(2):
        for jj in range(3):
            for kk in range(2):
                i = 0 + ii
                j_A = 2 + jj
                k = 0 + kk
                u_idx = i * m + j_A
                v_idx = j_A * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # C12 (2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    # A11*B12: 2x2 x 2x3 -> 2x3
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                i = 0 + ii
                j_A = 0 + jj
                k = 2 + kk
                u_idx = i * m + j_A
                v_idx = j_A * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # A12*B22: 2x3 x 3x3 -> 2x3
    for ii in range(2):
        for jj in range(3):
            for kk in range(3):
                i = 0 + ii
                j_A = 2 + jj
                k = 2 + kk
                u_idx = i * m + j_A
                v_idx = j_A * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # C21 (2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    strassen_block_2x2x2(2, 0, 0, 0, 2, 0)
    
    # A22*B21: 2x3 x 3x2 -> 2x2
    for ii in range(2):
        for jj in range(3):
            for kk in range(2):
                i = 2 + ii
                j_A = 2 + jj
                k = 0 + kk
                u_idx = i * m + j_A
                v_idx = j_A * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # C22 (2x3) = A21(2x2)*B12(2x3) + A22(2x3)*B22(3x3)
    # A21*B12: 2x2 x 2x3 -> 2x3
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                i = 2 + ii
                j_A = 0 + jj
                k = 2 + kk
                u_idx = i * m + j_A
                v_idx = j_A * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # A22*B22: 2x3 x 3x3 -> 2x3
    for ii in range(2):
        for jj in range(3):
            for kk in range(3):
                i = 2 + ii
                j_A = 2 + jj
                k = 2 + kk
                u_idx = i * m + j_A
                v_idx = j_A * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    rank = len(rank1_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (U_coeffs, V_coeffs, W_coeffs) in enumerate(rank1_terms):
        for idx, val in U_coeffs.items():
            U[idx][r] = val
        for idx, val in V_coeffs.items():
            V[idx][r] = val
        for idx, val in W_coeffs.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
