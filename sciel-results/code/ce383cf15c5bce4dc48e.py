import numpy as np

def construct():
    n, m, p = 3, 4, 4
    # Rank calculation:
    # 4 blocks of 2x2x2 (rank 7 each) = 28
    # 1x4x4 (rank 16) = 16
    # Total rank = 44
    rank = 44
    
    U = np.zeros((n * m, rank)) # 12 x 44
    V = np.zeros((m * p, rank)) # 16 x 44
    W = np.zeros((p * n, rank)) # 12 x 44
    
    # --- Part 1: 2x4x4 Decomposition (Rows 0-1 of A) ---
    # We use Strassen's algorithm for 2x2x2 blocks.
    # Local indices for 2x2 matrices:
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (local row-major for mapping logic, 
    #    but we must map to global W index: col * n + row)
    
    # Strassen factors for 2x2x2
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (a00 + a11) * (b00 + b11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # c00, c11
    
    # M2 = a11 * (b10 - b00)
    U_s[3, 1] = 1
    V_s[2, 1] = 1; V_s[0, 1] = -1
    W_s[2, 1] = 1; W_s[3, 1] = 1 # c10, c11
    
    # M3 = a00 * (b11 - b01)
    U_s[0, 2] = 1
    V_s[3, 2] = 1; V_s[1, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1 # c01, c11
    
    # M4 = (a10 + a00) * b00
    U_s[2, 3] = 1; U_s[0, 3] = 1
    V_s[0, 3] = 1
    W_s[0, 3] = 1; W_s[2, 3] = 1 # c00, c10
    
    # M5 = (a01 + a11) * b11
    U_s[1, 4] = 1; U_s[3, 4] = 1
    V_s[3, 4] = 1
    W_s[1, 4] = 1; W_s[3, 4] = 1 # c01, c11
    
    # M6 = (a00 - a10) * (b01 + b11)
    U_s[0, 5] = 1; U_s[2, 5] = -1
    V_s[1, 5] = 1; V_s[3, 5] = 1
    W_s[1, 5] = 1 # c01
    
    # M7 = (a11 - a01) * (b00 + b10)
    U_s[3, 6] = 1; U_s[1, 6] = -1
    V_s[0, 6] = 1; V_s[2, 6] = 1
    W_s[0, 6] = 1 # c00
    
    # Define the 4 blocks for the 2x4x4 part
    # Each block is a 2x2x2 multiplication.
    # Block (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
    blocks_2x4 = [
        (0, 0, 0, 0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 2, 0, 2), # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 0, 0, 0), # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 2, 0, 2), # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    ]
    
    col_offset = 0
    for (ar, ac, br, bc, cr, cc) in blocks_2x4:
        # Map local A indices (0-3) to global U indices
        # Local A index l = i*2 + j (i=0,1; j=0,1)
        # Global A row = ar + i, Global A col = ac + j
        # Global U index = (ar + i) * m + (ac + j)
        for i in range(2):
            for j in range(2):
                local_a_idx = i * 2 + j
                global_a_idx = (ar + i) * m + (ac + j)
                U[global_a_idx, col_offset:col_offset+7] += U_s[local_a_idx, :]
                
        # Map local B indices (0-3) to global V indices
        # Local B index l = i*2 + j
        # Global B row = br + i, Global B col = bc + j
        # Global V index = (br + i) * p + (bc + j)
        for i in range(2):
            for j in range(2):
                local_b_idx = i * 2 + j
                global_b_idx = (br + i) * p + (bc + j)
                V[global_b_idx, col_offset:col_offset+7] += V_s[local_b_idx, :]
                
        # Map local C indices (0-3) to global W indices
        # Local C index l = i*2 + j (i=row, j=col)
        # Global C row = cr + i, Global C col = cc + j
        # Global W index = (cc + j) * n + (cr + i)  <-- Note: W[k*n+i, r] means W[col*n+row, rank]
        for i in range(2):
            for j in range(2):
                local_c_idx = i * 2 + j
                global_c_row = cr + i
                global_c_col = cc + j
                global_w_idx = global_c_col * n + global_c_row
                W[global_w_idx, col_offset:col_offset+7] += W_s[local_c_idx, :]
        
        col_offset += 7
        
    # --- Part 2: 1x4x4 Decomposition (Row 2 of A) ---
    # A row 2 is 1x4. B is 4x4. C row 2 is 1x4.
    # C[2, j] = sum_k A[2, k] * B[k, j]
    # This is 4 independent dot products of length 4.
    # Each dot product has rank 4.
    # We iterate over each column j of C (0 to 3).
    # For each j, we have 4 terms corresponding to k=0..3.
    
    for j in range(4): # Column of C
        for k in range(4): # Summation index / Row of B
            # U part: A[2, k]
            global_a_idx = 2 * m + k # Row 2, Col k
            
            # V part: B[k, j]
            global_b_idx = k * p + j # Row k, Col j
            
            # W part: C[2, j]
            global_c_row = 2
            global_c_col = j
            global_w_idx = global_c_col * n + global_c_row # Col j, Row 2
            
            U[global_a_idx, col_offset] = 1
            V[global_b_idx, col_offset] = 1
            W[global_w_idx, col_offset] = 1
            
            col_offset += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
