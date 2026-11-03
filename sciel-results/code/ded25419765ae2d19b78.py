"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients for 2x2 blocks of 4x4 matrices
    # s is 0..6
    # alpha[s][block_row][block_col]
    alpha = [
        {(0,0): 1, (1,1): 1},      # s=0: A00 + A11
        {(1,0): 1, (1,1): 1},      # s=1: A10 + A11
        {(0,0): 1},                # s=2: A00
        {(1,1): 1},                # s=3: A11
        {(0,0): 1, (0,1): 1},      # s=4: A00 + A01
        {(1,0): 1, (0,0): -1},     # s=5: A10 - A00
        {(0,1): 1, (1,1): -1}      # s=6: A01 - A11
    ]
    
    # beta[s][block_row][block_col]
    beta = [
        {(0,0): 1, (1,1): 1},      # s=0: B00 + B11
        {(0,0): 1},                # s=1: B00
        {(0,1): 1, (1,1): -1},     # s=2: B01 - B11
        {(1,0): 1, (0,0): -1},     # s=3: B10 - B00
        {(1,1): 1},                # s=4: B11
        {(0,0): 1, (0,1): 1},      # s=5: B00 + B01
        {(1,0): 1, (1,1): 1}       # s=6: B10 + B11
    ]
    
    # delta[s][block_row][block_col]
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    delta = [
        {(0,0): 1, (1,1): 1},      # s=0: C00, C11
        {(1,0): 1, (1,1): -1},     # s=1: C10, C11
        {(0,1): 1, (1,1): 1},      # s=2: C01, C11
        {(0,0): 1, (1,0): 1},      # s=3: C00, C10
        {(0,0): -1, (0,1): 1},     # s=4: C00, C01
        {(1,1): 1},                # s=5: C11
        {(0,0): 1}                 # s=6: C00
    ]
    
    # Inner Strassen coefficients for 2x2 matrices
    # t is 0..6
    # inner_alpha[t][row][col]
    inner_alpha = [
        {(0,0): 1, (1,1): 1},      # t=0: X00 + X11
        {(1,0): 1, (1,1): 1},      # t=1: X10 + X11
        {(0,0): 1},                # t=2: X00
        {(1,1): 1},                # t=3: X11
        {(0,0): 1, (0,1): 1},      # t=4: X00 + X01
        {(1,0): 1, (0,0): -1},     # t=5: X10 - X00
        {(0,1): 1, (1,1): -1}      # t=6: X01 - X11
    ]
    
    # inner_beta[t][row][col]
    inner_beta = [
        {(0,0): 1, (1,1): 1},      # t=0: Y00 + Y11
        {(0,0): 1},                # t=1: Y00
        {(0,1): 1, (1,1): -1},     # t=2: Y01 - Y11
        {(1,0): 1, (0,0): -1},     # t=3: Y10 - Y00
        {(1,1): 1},                # t=4: Y11
        {(0,0): 1, (0,1): 1},      # t=5: Y00 + Y01
        {(1,0): 1, (1,1): 1}       # t=6: Y10 + Y11
    ]
    
    # inner_delta[t][row][col]
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    inner_delta = [
        {(0,0): 1, (1,1): 1},      # t=0: Z00, Z11
        {(1,0): 1, (1,1): -1},     # t=1: Z10, Z11
        {(0,1): 1, (1,1): 1},      # t=2: Z01, Z11
        {(0,0): 1, (1,0): 1},      # t=3: Z00, Z10
        {(0,0): -1, (0,1): 1},     # t=4: Z00, Z01
        {(1,1): 1},                # t=5: Z11
        {(0,0): 1}                 # t=6: Z00
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Calculate contributions to U, V, W
            # U: A entries
            # u_st is linear combo of L_s entries
            # L_s entries are linear combos of A_block entries
            # L_s[i,j] = sum_{p,q} alpha[s][p,q] A_block[p,q][i,j]
            # u_st = sum_{i,j} inner_alpha[t][i,j] L_s[i,j]
            # u_st = sum_{p,q,i,j} inner_alpha[t][i,j] alpha[s][p,q] A[2p+i, 2q+j]
            
            for (i, j), coeff_inner_u in inner_alpha[t].items():
                for (p, q), coeff_outer_u in alpha[s].items():
                    # A index: row = 2*p + i, col = 2*q + j
                    a_row = 2 * p + i
                    a_col = 2 * q + j
                    a_idx = a_row * m + a_col
                    U[a_idx][r] += coeff_inner_u * coeff_outer_u
            
            # V: B entries
            # v_st is linear combo of R_s entries
            # R_s[i,j] = sum_{p,q} beta[s][p,q] B_block[p,q][i,j]
            # v_st = sum_{i,j} inner_beta[t][i,j] R_s[i,j]
            # v_st = sum_{p,q,i,j} inner_beta[t][i,j] beta[s][p,q] B[2p+i, 2q+j]
            
            for (i, j), coeff_inner_v in inner_beta[t].items():
                for (p, q), coeff_outer_v in beta[s].items():
                    # B index: row = 2*p + i, col = 2*q + j
                    # Note: B is m x p. m=4, p=4.
                    # Block p,q corresponds to rows 2p..2p+1, cols 2q..2q+1?
                    # In matrix multiplication A (n x m) * B (m x p).
                    # A is 4x4. B is 4x4.
                    # Block row p, col q of B.
                    # B index: row = 2*p + i, col = 2*q + j
                    b_row = 2 * p + i
                    b_col = 2 * q + j
                    b_idx = b_row * p + b_col
                    V[b_idx][r] += coeff_inner_v * coeff_outer_v
            
            # W: C entries
            # Contribution to C_block[uv][i,j] is delta[s][u,v] * inner_delta[t][i,j] * u_st * v_st
            # C global row = 2*u + i
            # C global col = 2*v + j
            # W index convention: k*n + i where k is col, i is row.
            # So W index = (2*v + j) * n + (2*u + i)
            
            for (u, v), coeff_outer_w in delta[s].items():
                for (i, j), coeff_inner_w in inner_delta[t].items():
                    c_row = 2 * u + i
                    c_col = 2 * v + j
                    w_idx = c_col * n + c_row
                    W[w_idx][r] += coeff_outer_w * coeff_inner_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
