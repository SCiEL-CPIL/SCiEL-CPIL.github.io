"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 using Strassen's algorithm on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Each entry is (a_coeffs, b_coeffs, c_coeffs)
    # Keys are local (row, col) indices
    steps = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B01-B00)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A10)B11
        ({(0,0):1, (1,0):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B10)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # M7 = (A01-A11)(B11+B01)
        ({(0,1):1, (1,1):-1}, {(1,1):1, (0,1):1}, {(0,0):1, (0,1):1})
    ]
    
    # Block mappings for 2x4 x 4x4 multiplication
    # (a_col_off, b_row_off, b_col_off, c_row_off)
    # a_row_off is always 0, c_col_off is always 0
    # A is 2x4, split into A1 (cols 0-1) and A2 (cols 2-3)
    # B is 4x4, split into B11 (r0-1,c0-1), B21 (r2-3,c0-1), B12 (r0-1,c2-3), B22 (r2-3,c2-3)
    # C is 2x4, split into C1 (cols 0-1) and C2 (cols 2-3)
    
    blocks = [
        # A1, B11 -> C1
        (0, 0, 0, 0),
        # A2, B21 -> C1
        (2, 2, 0, 0),
        # A1, B12 -> C2
        (0, 0, 2, 2),
        # A2, B22 -> C2
        (2, 2, 2, 2)
    ]
    
    r = 0
    for block in blocks:
        a_c_off, b_r_off, b_c_off, c_r_off = block
        for step in steps:
            a_coeffs, b_coeffs, c_coeffs = step
            
            # Fill U (A indices)
            for (i, j), val in a_coeffs.items():
                gi = i
                gj = j + a_c_off
                U[gi * m + gj][r] = val
                
            # Fill V (B indices)
            for (j, k), val in b_coeffs.items():
                gj = j + b_r_off
                gk = k + b_c_off
                V[gj * p + gk][r] = val
                
            # Fill W (C indices)
            # W index is k*n + i
            # Local C indices are (k_loc, i_loc)
            for (k, i), val in c_coeffs.items():
                gk = k + c_r_off
                gi = i
                W[gk * n + gi][r] = val
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
