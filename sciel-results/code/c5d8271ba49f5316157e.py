"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for 2x2 by 2x2
    # Local A indices: 00, 01, 10, 11
    # Local B indices: 00, 01, 10, 11
    # Local C indices: 00, 01, 10, 11
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # List of (U_coeffs, V_coeffs, W_coeffs) for each of the 7 terms
    # U_coeffs: dict local_A_idx -> coeff
    # V_coeffs: dict local_B_idx -> coeff
    # W_coeffs: dict local_C_idx -> coeff
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # A:00,11; B:00,11; C:00,11
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),             # A:10,11; B:00; C:10
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),            # A:00; B:01,-11; C:01
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),      # A:11; B:10,-00; C:00,10
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),      # A:00,01; B:11; C:-00,01
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # A:10,-00; B:00,01; C:11
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),      # A:01,-11; B:10,11; C:00
    ]
    
    # Define the 4 blocks
    # (A_col_offset, B_row_offset, B_col_offset)
    # A cols 0-1 or 2-3. B rows 0-1 or 2-3. B cols 0-1 or 2-3.
    blocks = [
        (0, 0, 0), # A_L, B_TL
        (0, 0, 2), # A_L, B_TR
        (2, 2, 0), # A_R, B_BL
        (2, 2, 2), # A_R, B_BR
    ]
    
    for a_off, b_row_off, b_col_off in blocks:
        # Map local indices to global indices
        # Local A: r in {0,1}, c in {0,1} -> Global i = r, j = c + a_off
        # Global U index: i * m + j
        def local_a_to_global(r, c):
            i = r
            j = c + a_off
            return i * m + j
            
        # Local B: r in {0,1}, c in {0,1} -> Global j = r + b_row_off, k = c + b_col_off
        # Global V index: j * p + k
        def local_b_to_global(r, c):
            j = r + b_row_off
            k = c + b_col_off
            return j * p + k
            
        # Local C: r in {0,1}, c in {0,1} -> Global i = r, k = c + b_col_off
        # Global W index: k * n + i
        def local_c_to_global(r, c):
            i = r
            k = c + b_col_off
            return k * n + i

        for u_coeff, v_coeff, w_coeff in strassen_terms:
            # Fill U
            for loc_idx, val in u_coeff.items():
                r, c = divmod(loc_idx, 2) # 0:0,0 1:0,1 2:1,0 3:1,1
                g_idx = local_a_to_global(r, c)
                U[g_idx][col] += val
                
            # Fill V
            for loc_idx, val in v_coeff.items():
                r, c = divmod(loc_idx, 2)
                g_idx = local_b_to_global(r, c)
                V[g_idx][col] += val
                
            # Fill W
            for loc_idx, val in w_coeff.items():
                r, c = divmod(loc_idx, 2)
                g_idx = local_c_to_global(r, c)
                W[g_idx][col] += val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
