"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen steps for 4x4 (Block level)
    # Each step is (A_comb, B_comb) where A_comb is a 2x2 matrix of linear forms
    # Linear forms are dicts: {index_in_16: coeff}
    
    def make_block_comb(blocks):
        # blocks is a list of (block_idx, coeff)
        # block_idx 0: 11, 1: 12, 2: 21, 3: 22
        # Returns a 2x2 list of dicts
        res = [[{} for _ in range(2)] for _ in range(2)]
        for b_idx, coeff in blocks:
            # b_idx maps to (row_block, col_block)
            rb = b_idx // 2
            cb = b_idx % 2
            for r in range(2):
                for c in range(2):
                    global_idx = (rb * 2 + r) * 4 + (cb * 2 + c)
                    if global_idx not in res[r][c]:
                        res[r][c][global_idx] = 0
                    res[r][c][global_idx] += coeff
        return res

    # Strassen 4x4 steps
    # M1 = (A11+A22)(B11+B22)
    steps_4x4 = [
        (make_block_comb([(0, 1), (3, 1)]), make_block_comb([(0, 1), (3, 1)])), # M1
        (make_block_comb([(2, 1), (3, 1)]), make_block_comb([(0, 1)])),          # M2
        (make_block_comb([(0, 1)]), make_block_comb([(1, 1), (3, -1)])),         # M3
        (make_block_comb([(3, 1)]), make_block_comb([(2, 1), (0, -1)])),         # M4
        (make_block_comb([(0, 1), (1, 1)]), make_block_comb([(3, 1)])),          # M5
        (make_block_comb([(2, 1), (0, -1)]), make_block_comb([(0, 1), (1, 1)])), # M6
        (make_block_comb([(1, 1), (3, -1)]), make_block_comb([(2, 1), (3, 1)]))  # M7
    ]
    
    # Strassen 2x2 steps
    # For X, Y 2x2 matrices (entries 0,1,2,3 corresponding to 00,01,10,11)
    # T1 = (X00+X11)(Y00+Y11)
    # T2 = (X10+X11)Y00
    # T3 = X00(Y01-Y11)
    # T4 = X11(Y10-Y00)
    # T5 = (X00+X01)Y11
    # T6 = (X10-X00)(Y00+Y01)
    # T7 = (X01-X11)(Y10+Y11)
    
    def make_2x2_comb(indices_coeffs):
        # indices_coeffs: list of (entry_idx, coeff)
        # Returns dict {entry_idx: coeff}
        d = {}
        for idx, c in indices_coeffs:
            if idx not in d:
                d[idx] = 0
            d[idx] += c
        return d

    steps_2x2 = [
        (make_2x2_comb([(0, 1), (3, 1)]), make_2x2_comb([(0, 1), (3, 1)])), # T1
        (make_2x2_comb([(2, 1), (3, 1)]), make_2x2_comb([(0, 1)])),          # T2
        (make_2x2_comb([(0, 1)]), make_2x2_comb([(1, 1), (3, -1)])),         # T3
        (make_2x2_comb([(3, 1)]), make_2x2_comb([(2, 1), (0, -1)])),         # T4
        (make_2x2_comb([(0, 1), (1, 1)]), make_2x2_comb([(3, 1)])),          # T5
        (make_2x2_comb([(2, 1), (0, -1)]), make_2x2_comb([(0, 1), (1, 1)])), # T6
        (make_2x2_comb([(1, 1), (3, -1)]), make_2x2_comb([(2, 1), (3, 1)]))  # T7
    ]
    
    # Reconstruction coefficients for 2x2 result Z
    # Z00 = T1 + T4 - T5 + T7
    # Z01 = T3 + T5
    # Z10 = T2 + T4
    # Z11 = T1 - T2 + T3 + T6
    # Z_idx: 0->00, 1->01, 2->10, 3->11
    # T_idx: 0->T1 ... 6->T7
    recon_2x2 = [
        [1, 0, 0, 1, -1, 0, 1], # Z00 coeffs for T1..T7
        [0, 1, 1, 0, 0, 0, 0],  # Z01
        [0, 0, 1, 1, 0, 0, 0],  # Z10
        [1, -1, 1, 0, 0, 1, 0]  # Z11
    ]
    
    # Block placement in 4x4 result C
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # M_idx: 0->M1 ... 6->M7
    # C_block_idx: 0->11, 1->12, 2->21, 3->22
    recon_4x4 = [
        [1, 0, 0, 1, -1, 0, 1], # C11 coeffs for M1..M7
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]

    for i in range(7):
        A_block_comb, B_block_comb = steps_4x4[i]
        for s in range(7):
            X_comb, Y_comb = steps_2x2[s]
            
            # Construct U vector (16 entries)
            for r in range(2):
                for c in range(2):
                    for g_idx, coeff in A_block_comb[r][c].items():
                        local_idx = r * 2 + c
                        if local_idx in X_comb:
                            x_coeff = X_comb[local_idx]
                            total_coeff = coeff * x_coeff
                            if total_coeff != 0:
                                U[g_idx][col] += total_coeff
            
            # Construct V vector (16 entries)
            for r in range(2):
                for c in range(2):
                    for g_idx, coeff in B_block_comb[r][c].items():
                        local_idx = r * 2 + c
                        if local_idx in Y_comb:
                            y_coeff = Y_comb[local_idx]
                            total_coeff = coeff * y_coeff
                            if total_coeff != 0:
                                V[g_idx][col] += total_coeff
            
            # Construct W vector (16 entries)
            for b_idx in range(4):
                c_b = recon_4x4[b_idx][i]
                if c_b == 0:
                    continue
                br = b_idx // 2
                bc = b_idx % 2
                
                for x in range(2):
                    for y in range(2):
                        xy_idx = x * 2 + y
                        c_xy_s = recon_2x2[xy_idx][s]
                        if c_xy_s == 0:
                            continue
                        total_w_coeff = c_b * c_xy_s
                        if total_w_coeff == 0:
                            continue
                        
                        c_row = 2 * br + x
                        c_col = 2 * bc + y
                        w_idx = c_col * 4 + c_row
                        W[w_idx][col] += total_w_coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
