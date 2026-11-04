import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # Indices for 2x2 matrix: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Coefficients for U (A matrix)
    # x00, x01, x10, x11
    # M1: x00=1, x11=1
    # M2: x10=1, x11=1
    # M3: x00=1
    # M4: x11=1
    # M5: x00=1, x01=1
    # M6: x10=1, x00=-1
    # M7: x01=1, x11=-1
    
    u_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Coefficients for V (B matrix)
    # y00, y01, y10, y11
    # M1: y00=1, y11=1
    # M2: y00=1
    # M3: y01=1, y11=-1
    # M4: y10=1, y00=-1
    # M5: y11=1
    # M6: y00=1, y01=1
    # M7: y10=1, y11=1
    
    v_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # Coefficients for W (C matrix)
    # z00, z01, z10, z11
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    w_coeffs = [
        {0: 1, 3: 1, 2: -1, 1: 1},   # z00 (M1, M4, -M5, M7) -> indices 0,3,2,1 in M list? No, M1..M7
        # Let's map M1..M7 to 0..6
        # z00: M1(0) + M4(3) - M5(4) + M7(6)
        # z01: M3(2) + M5(4)
        # z10: M2(1) + M4(3)
        # z11: M1(0) - M2(1) + M3(2) + M6(5)
    ]
    
    # Correct W coefficients mapping M1..M7 to 0..6
    w_terms = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # z00
        {2: 1, 4: 1},                 # z01
        {1: 1, 3: 1},                 # z10
        {0: 1, 1: -1, 2: 1, 5: 1}    # z11
    ]
    
    # Define the 4 blocks
    # Block 1: A11 * B11 -> C1 (part 1)
    # A11: rows 0-1, cols 0-1 of A
    # B11: rows 0-1, cols 0-1 of B
    # C1: rows 0-1, cols 0-1 of C
    
    # Block 2: A12 * B21 -> C1 (part 2)
    # A12: rows 0-1, cols 2-3 of A
    # B21: rows 2-3, cols 0-1 of B
    # C1: rows 0-1, cols 0-1 of C
    
    # Block 3: A11 * B12 -> C2 (part 1)
    # A11: rows 0-1, cols 0-1 of A
    # B12: rows 0-1, cols 2-3 of B
    # C2: rows 0-1, cols 2-3 of C
    
    # Block 4: A12 * B22 -> C2 (part 2)
    # A12: rows 0-1, cols 2-3 of A
    # B22: rows 2-3, cols 2-3 of B
    # C2: rows 0-1, cols 2-3 of C
    
    blocks = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        (0, 0, 0, 0, 0, 0),   # A11 * B11 -> C1
        (0, 2, 2, 0, 0, 0),   # A12 * B21 -> C1
        (0, 0, 0, 2, 0, 2),   # A11 * B12 -> C2
        (0, 2, 2, 2, 0, 2)    # A12 * B22 -> C2
    ]
    
    col_idx = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off in blocks:
        for m_idx in range(7):
            # Fill U
            for local_i, coeff in u_coeffs[m_idx].items():
                global_i = a_r_off + local_i
                global_j = a_c_off + local_i # Wait, local_i is row index for A?
                # A is 2x4.
                # local indices 0,1 for rows, 0,1 for cols within 2x2 block.
                # u_coeffs keys are local row indices?
                # Let's re-verify u_coeffs.
                # M1: (x00+x11). x00 is row 0, col 0. x11 is row 1, col 1.
                # So keys should be (row, col) pairs.
                pass
                
            # Let's redefine u_coeffs with (row, col) keys
            pass
            
    # Redefine coefficients with (row, col)
    u_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    
    v_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    
    # W coefficients: M1..M7 -> 0..6
    # z00: M1 + M4 - M5 + M7
    # z01: M3 + M5
    # z10: M2 + M4
    # z11: M1 - M2 + M3 + M6
    w_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # z00
        {2: 1, 4: 1},                 # z01
        {1: 1, 3: 1},                 # z10
        {0: 1, 1: -1, 2: 1, 5: 1}    # z11
    ]
    
    col_idx = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off in blocks:
        for m_idx in range(7):
            # Fill U
            for (lr, lc), coeff in u_coeffs[m_idx].items():
                global_i = a_r_off + lr
                global_j = a_c_off + lc
                U[global_i * m + global_j][col_idx] = coeff
                
            # Fill V
            for (lr, lc), coeff in v_coeffs[m_idx].items():
                global_j = b_r_off + lr
                global_k = b_c_off + lc
                V[global_j * p + global_k][col_idx] = coeff
                
            # Fill W
            # W[k*n + i, c]
            # local z indices: (0,0)->z00, (0,1)->z01, (1,0)->z10, (1,1)->z11
            # w_coeffs keys are M indices (0..6)
            # We need to add contributions to W for each z_ij
            for z_local_idx, w_term in enumerate(w_coeffs):
                # z_local_idx 0: z00 (row 0, col 0)
                # z_local_idx 1: z01 (row 0, col 1)
                # z_local_idx 2: z10 (row 1, col 0)
                # z_local_idx 3: z11 (row 1, col 1)
                zr = z_local_idx // 2
                zc = z_local_idx % 2
                
                coeff = w_term.get(m_idx, 0)
                if coeff != 0:
                    global_i = c_r_off + zr
                    global_k = c_c_off + zc
                    W[global_k * n + global_i][col_idx] = coeff
                    
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
