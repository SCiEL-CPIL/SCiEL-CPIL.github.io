import time

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # U indices for A00, A01, A10, A11
    # V indices for B00, B01, B10, B11
    # W indices for C00, C01, C10, C11
    
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B01-B11)
    # P4: A11(B10-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [1, 0, -1, 1, 0, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 1]   # B11
    ]
    
    strassen_W = [
        [1, 0, 0, 0, 0, 0, 1],  # C00 = P1 - P4 - P5 + P7
        [0, 0, 1, 0, 1, 0, 0],  # C01 = P3 + P5
        [0, 1, 0, 1, 0, 0, 0],  # C10 = P2 + P4
        [1, -1, -1, 0, 0, 1, 0] # C11 = P1 - P2 - P3 + P6
    ]
    
    # Row 2 additions
    # P8: A20 B00
    # P9: A21 B10
    # P10: A20 B01
    # P11: A21 B11
    
    row2_U = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], # A20
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # A21
    ]
    
    row2_V = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # B00
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # B01
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # B10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # B11
    ]
    
    row2_W = [
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # C20 = P8 + P9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # C21 = P10 + P11
    ]
    
    # Combine into 11x? matrices for the 3x2x2 block
    # U_block: 6x11 (3 rows x 2 cols)
    # V_block: 4x11 (2 rows x 2 cols)
    # W_block: 6x11 (2 cols x 3 rows)
    
    U_block = [row for row in strassen_U] + row2_U
    V_block = [row for row in strassen_V] + row2_V
    W_block = [row for row in strassen_W] + row2_W
    
    # Now map to global indices
    # 2 blocks of B (top rows 0,1 and bottom rows 2,3)
    # 3 sub-blocks of B (cols 0-1, 2-3, 4-5)
    
    for b_block in range(2): # 0: top, 1: bottom
        for k_block in range(3): # 0: left, 1: mid, 2: right
            
            # Global indices for B
            b_row_offset = b_block * 2
            b_col_offset = k_block * 2
            
            # Global indices for A (same for all)
            a_row_offset = 0
            a_col_offset = 0 # A is 3x4, we use cols 0,1 for top B? 
            # Wait, A is 3x4. B is 4x6.
            # C = A B.
            # C_ik = sum_j A_ij B_jk.
            # We split B into B_top (rows 0,1) and B_bot (rows 2,3).
            # C = A_left B_top + A_right B_bot?
            # No, A is 3x4.
            # A = [A_1 | A_2] where A_1 is 3x2 (cols 0,1), A_2 is 3x2 (cols 2,3).
            # B = [B_top; B_bot] where B_top is 2x6, B_bot is 2x6.
            # C = A_1 B_top + A_2 B_bot.
            
            # So for b_block 0 (top), we use A_1 (cols 0,1).
            # For b_block 1 (bottom), we use A_2 (cols 2,3).
            
            a_col_offset = b_block * 2
            
            # Global indices for C
            c_col_offset = k_block * 2
            c_row_offset = 0
            
            # Fill U
            for i in range(3): # A rows
                for j in range(2): # A cols (local 0,1)
                    global_u_idx = i * 4 + (a_col_offset + j)
                    local_u_idx = i * 2 + j
                    for c in range(11):
                        U[global_u_idx][col + c] = U_block[local_u_idx][c]
                        
            # Fill V
            for j in range(2): # B rows (local 0,1)
                for k in range(2): # B cols (local 0,1)
                    global_v_idx = (b_row_offset + j) * 6 + (b_col_offset + k)
                    local_v_idx = j * 2 + k
                    for c in range(11):
                        V[global_v_idx][col + c] = V_block[local_v_idx][c]
                        
            # Fill W
            for k in range(2): # C cols (local 0,1)
                for i in range(3): # C rows
                    global_w_idx = (c_col_offset + k) * 3 + i
                    local_w_idx = k * 3 + i
                    for c in range(11):
                        W[global_w_idx][col + c] = W_block[local_w_idx][c]
                        
            col += 11
            
    return {"U": U, "V": V, "W": W}

