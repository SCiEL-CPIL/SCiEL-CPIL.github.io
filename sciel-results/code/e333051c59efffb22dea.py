import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen steps
    # Each step is a list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local (r,c) to coeff
    # v_coeffs: dict mapping local (r,c) to coeff
    # w_coeffs: dict mapping local (r,c) to coeff
    
    # Local indices 0..1 for 2x2 matrices
    # M0 = (X00 + X11)(Y00 + Y11)
    # M1 = X10(Y00 + Y01)
    # M2 = (X00 + X01)Y10
    # M3 = X11(Y10 + Y11)
    # M4 = (X00 - X11)(Y01 - Y11)
    # M5 = (X01 - X11)(Y00 + Y10)
    # M6 = (X00 - X10)(Y01 + Y11)
    
    strassen_steps = [
        # M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # Z00+=, Z11+=
        # M1
        ({(1,0): 1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): -1}), # Z10+=, Z11-=
        # M2
        ({(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,1): 1, (1,1): 1}), # Z01+=, Z11+=
        # M3
        ({(1,1): 1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}), # Z00+=, Z10+=
        # M4
        ({(0,0): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1}), # Z00-=, Z01+=
        # M5
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}), # Z11+=
        # M6
        ({(0,0): 1, (1,0): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}), # Z00+=, Z01+=
    ]
    
    # Wait, let's re-verify the Z coefficients for each M.
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    
    # Let's re-define w_coeffs based on this.
    # M0: Z00+=1, Z11+=1
    # M1: Z10+=1, Z11-=1
    # M2: Z01+=1, Z11+=1
    # M3: Z00+=1, Z10+=1
    # M4: Z00-=1, Z01+=1
    # M5: Z11+=1
    # M6: Z00+=1, Z01+=1
    
    strassen_steps = [
        # M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1
        ({(1,0): 1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): -1}),
        # M2
        ({(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,1): 1, (1,1): 1}),
        # M3
        ({(1,1): 1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}),
        # M4
        ({(0,0): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1}),
        # M5
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M6
        ({(0,0): 1, (1,0): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
    ]

    # Blocks: (a_start_row, a_start_col, b_start_row, b_start_col, c_start_row, c_start_col)
    # A is 2x4. B is 4x4. C is 4x2 (stored as k*n+i).
    # Block 1: A11 B11 -> C11
    # A11: rows 0-1, cols 0-1. Start (0,0).
    # B11: rows 0-1, cols 0-1. Start (0,0).
    # C11: rows 0-1 (k), cols 0-1 (i). Start (0,0) in C space (k,i).
    # Block 2: A12 B21 -> C11
    # A12: rows 0-1, cols 2-3. Start (0,2).
    # B21: rows 2-3, cols 0-1. Start (2,0).
    # C11: rows 0-1, cols 0-1. Start (0,0).
    # Block 3: A11 B12 -> C12
    # A11: rows 0-1, cols 0-1. Start (0,0).
    # B12: rows 0-1, cols 2-3. Start (0,2).
    # C12: rows 2-3, cols 0-1. Start (2,0).
    # Block 4: A12 B22 -> C12
    # A12: rows 0-1, cols 2-3. Start (0,2).
    # B22: rows 2-3, cols 2-3. Start (2,2).
    # C12: rows 2-3, cols 0-1. Start (2,0).

    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 2, 0),
        (0, 2, 2, 2, 2, 0),
    ]

    for ar, ac, br, bc, cr, cc in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            # Fill U
            for (lr, lc), coeff in u_coeffs.items():
                global_r = ar + lr
                global_c = ac + lc
                idx = global_r * m + global_c
                U[idx][col] += coeff
            
            # Fill V
            for (lr, lc), coeff in v_coeffs.items():
                global_r = br + lr
                global_c = bc + lc
                idx = global_r * p + global_c
                V[idx][col] += coeff
            
            # Fill W
            for (lr, lc), coeff in w_coeffs.items():
                # C is 4x2. lr is local row (k), lc is local col (i).
                global_r = cr + lr
                global_c = cc + lc
                idx = global_r * n + global_c
                W[idx][col] += coeff
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
