import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2 multiplication
    # X is 2x2, Y is 2x2
    # M1 = X00 * (Y00 + Y11)
    # M2 = (X00 + X10) * Y11
    # M3 = (X01 + X11) * Y00
    # M4 = X11 * (Y00 - Y10)
    # M5 = (X00 + X01) * Y10
    # M6 = (X10 - X00) * (Y01 + Y11)
    # M7 = (X01 - X11) * (Y00 + Y10)
    
    # Z00 = M1 + M4 + M5 - M7
    # Z01 = M3 + M7
    # Z10 = M2 + M6
    # Z11 = M1 + M3 + M4 - M6
    
    # We define the local indices for X and Y
    # X indices: (i, j) local 0..1
    # Y indices: (j, k) local 0..1
    # Z indices: (i, k) local 0..1
    
    # Strassen terms definition:
    # term_id: (U_coeff_dict, V_coeff_dict, W_coeff_dict)
    # U_coeff: { (i,j): coeff }
    # V_coeff: { (j,k): coeff }
    # W_coeff: { (i,k): coeff }
    
    strassen_terms = [
        # M1
        ( {(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1} ), # Wait, Z11 also has M1
        # M2
        ( {(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1} ),
        # M3
        ( {(0,1): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): 1} ),
        # M4
        ( {(1,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1, (1,1): 1} ),
        # M5
        ( {(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,0): 1} ),
        # M6
        ( {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}, {(1,0): -1, (1,1): -1} ), # Z10=M2+M6, Z11=M1+M3+M4-M6
        # M7
        ( {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): -1, (0,1): 1} ),
    ]
    
    # Wait, I need to be precise with the W coefficients.
    # Z00 = M1 + M4 + M5 - M7
    # Z01 = M3 + M7
    # Z10 = M2 + M6
    # Z11 = M1 + M3 + M4 - M6
    
    # Let's re-verify M6 and M7 signs in Z.
    # Z11 = M1 + M3 + M4 - M6. So M6 coeff is -1.
    # Z10 = M2 + M6. So M6 coeff is 1.
    # M7: Z00 has -M7. Z01 has +M7.
    
    # Corrected W coefficients:
    # M1: Z00: 1, Z11: 1
    # M2: Z10: 1
    # M3: Z01: 1, Z11: 1
    # M4: Z00: 1, Z11: 1
    # M5: Z00: 1
    # M6: Z10: 1, Z11: -1
    # M7: Z00: -1, Z01: 1
    
    strassen_terms = [
        ( {(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1} ),
        ( {(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1} ),
        ( {(0,1): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): 1} ),
        ( {(1,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1, (1,1): 1} ),
        ( {(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,0): 1} ),
        ( {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): -1} ),
        ( {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): -1, (0,1): 1} ),
    ]
    
    # Define the 4 blocks
    # Block 1: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1) -> C cols 0,1
    # Block 2: A2 (cols 2,3) * B21 (rows 2,3; cols 0,1) -> C cols 0,1
    # Block 3: A1 (cols 0,1) * B12 (rows 0,1; cols 2,3) -> C cols 2,3
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3; cols 2,3) -> C cols 2,3
    
    blocks = [
        { 'a_col_offset': 0, 'b_row_offset': 0, 'b_col_offset': 0, 'c_col_offset': 0 },
        { 'a_col_offset': 2, 'b_row_offset': 2, 'b_col_offset': 0, 'c_col_offset': 0 },
        { 'a_col_offset': 0, 'b_row_offset': 0, 'b_col_offset': 2, 'c_col_offset': 2 },
        { 'a_col_offset': 2, 'b_row_offset': 2, 'b_col_offset': 2, 'c_col_offset': 2 },
    ]
    
    for block in blocks:
        a_off = block['a_col_offset']
        b_r_off = block['b_row_offset']
        b_c_off = block['b_col_offset']
        c_c_off = block['c_col_offset']
        
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            for (li, lj), uc in u_coeffs.items():
                # Local i, j. Global i, j
                gi = li
                gj = lj + a_off
                U[gi * m + gj][col] += uc
            
            for (lj, lk), vc in v_coeffs.items():
                # Local j, k. Global j, k
                gj = lj + b_r_off
                gk = lk + b_c_off
                V[gj * p + gk][col] += vc
            
            for (li, lk), wc in w_coeffs.items():
                # Local i, k. Global i, k
                gi = li
                gk = lk + c_c_off
                # W index is k * n + i
                W[gk * n + gi][col] += wc
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
