"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Each entry is (U_dict, V_dict, W_dict)
    # Keys are (local_row, local_col)
    strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}) # M7
    ]
    
    # Blocks: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    # A is 2x4, B is 4x4, C is 2x4
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        for s_idx, (U_loc, V_loc, W_loc) in enumerate(strassen):
            # Process U
            for (lu, lv), coeff in U_loc.items():
                gi = ar + lu
                gj = ac + lv
                U[gi * m + gj][col] = coeff
            
            # Process V
            for (lu, lv), coeff in V_loc.items():
                gj = br + lu
                gk = bc + lv
                V[gj * p + gk][col] = coeff
            
            # Process W
            # W index is k*n + i
            # Local C indices are (lu, lv) where lu is row (i) and lv is col (k)
            # So global i = cr + lu, global k = cc + lv
            for (lu, lv), coeff in W_loc.items():
                gi = cr + lu
                gk = cc + lv
                W[gk * n + gi][col] = coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
