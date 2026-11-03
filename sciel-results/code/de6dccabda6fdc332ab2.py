"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84  # 3 blocks * 4 Strassen instances * 7 multipliers
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen multipliers
    # m1: (a+d)(e+h)
    # m2: (c+d)e
    # m3: a(f-h)
    # m4: d(g-e)
    # m5: (a+b)h
    # m6: (c-a)(e+f)
    # m7: (b-d)(g+h)
    
    # U indices for A (2x2): 0,0; 0,1; 1,0; 1,1
    # V indices for B (2x2): 0,0; 0,1; 1,0; 1,1
    # W indices for C (2x2): 0,0; 0,1; 1,0; 1,1
    
    strassen_defs = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [a, b, c, d] -> indices 0, 1, 2, 3
        # V: [e, f, g, h] -> indices 0, 1, 2, 3
        # W: [c11, c12, c21, c22] -> indices 0, 1, 2, 3
        
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # m5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # m6
        ([ -1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Loop over the 3 blocks of B (cols 0-1, 2-3, 4-5)
    for block in range(3):
        b_col_offset = block * 2
        
        # 4 Strassen instances per block
        # 1. A11 * Bk1 -> C_top (rows 0-1 of C)
        # 2. A12 * Bk2 -> C_top
        # 3. A21 * Bk1 -> C_bot (rows 2-3 of C)
        # 4. A22 * Bk2 -> C_bot
        
        instances = [
            # (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
            (0, 0, 0, b_col_offset, 0, b_col_offset),
            (0, 2, 2, b_col_offset, 0, b_col_offset),
            (2, 0, 0, b_col_offset, 2, b_col_offset),
            (2, 2, 2, b_col_offset, 2, b_col_offset),
        ]
        
        for A_r, A_c, B_r, B_c, C_r, C_c in instances:
            for u_coeffs, v_coeffs, w_coeffs in strassen_defs:
                # Add to U
                for local_r in range(2):
                    for local_c in range(2):
                        idx_u = (A_r + local_r) * m + (A_c + local_c)
                        U[idx_u][col] += u_coeffs[local_r * 2 + local_c]
                
                # Add to V
                for local_r in range(2):
                    for local_c in range(2):
                        idx_v = (B_r + local_r) * p + (B_c + local_c)
                        V[idx_v][col] += v_coeffs[local_r * 2 + local_c]
                
                # Add to W
                for local_r in range(2):
                    for local_c in range(2):
                        idx_w = (C_c + local_c) * n + (C_r + local_r)
                        W[idx_w][col] += w_coeffs[local_r * 2 + local_c]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
