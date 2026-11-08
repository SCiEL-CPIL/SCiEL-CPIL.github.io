import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # Inner Strassen for 2x2x2
    # U, V, W are 4x7
    U_inner = np.zeros((4, 7))
    V_inner = np.zeros((4, 7))
    W_inner = np.zeros((4, 7))
    
    # M1
    U_inner[0, 0] = 1; U_inner[3, 0] = 1
    V_inner[0, 0] = 1; V_inner[3, 0] = 1
    W_inner[0, 0] = 1; W_inner[3, 0] = 1
    
    # M2
    U_inner[2, 1] = 1; U_inner[3, 1] = 1
    V_inner[0, 1] = 1
    W_inner[2, 1] = 1; W_inner[3, 1] = -1
    
    # M3
    U_inner[0, 2] = 1
    V_inner[1, 2] = 1; V_inner[3, 2] = -1
    W_inner[1, 2] = 1; W_inner[3, 2] = 1
    
    # M4
    U_inner[3, 3] = 1
    V_inner[0, 3] = -1; V_inner[2, 3] = 1
    W_inner[0, 3] = 1; W_inner[2, 3] = 1
    
    # M5
    U_inner[0, 4] = 1; U_inner[1, 4] = 1
    V_inner[3, 4] = 1
    W_inner[0, 4] = -1; W_inner[1, 4] = 1
    
    # M6
    U_inner[0, 5] = -1; U_inner[2, 5] = 1
    V_inner[0, 5] = 1; V_inner[1, 5] = 1
    W_inner[3, 5] = 1
    
    # M7
    U_inner[1, 6] = 1; U_inner[3, 6] = -1
    V_inner[2, 6] = 1; V_inner[3, 6] = 1
    W_inner[0, 6] = 1
    
    # Outer Strassen for 4x4x4
    # 7 steps
    # Each step involves 4 blocks of A and 4 blocks of B
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Define block indices in 4x4 (row-major)
    # 00: 0,1,4,5
    # 01: 2,3,6,7
    # 10: 8,9,12,13
    # 11: 10,11,14,15
    
    block_indices_A = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    block_indices_B = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # Outer Strassen coefficients
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    outer_coeffs = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, {0: 1},       # M2
        {0: 1}, {1: 1, 3: -1},      # M3
        {3: 1}, {2: 1, 0: -1},      # M4
        {0: 1, 1: 1}, {3: 1},       # M5
        {2: 1, 0: -1}, {0: 1, 1: 1}, # M6
        {1: 1, 3: -1}, {2: 1, 3: 1}  # M7
    ]
    
    # W coefficients for outer Strassen
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    outer_W_coeffs = {
        0: [1, 0, 0, 1], # C00 gets M1
        1: [0, 1, 0, 0], # C01 gets M3? No.
    }
    
    # Let's just accumulate W
    # W_outer is 16x7? No.
    # For each outer step k, we have an inner decomposition.
    # The inner W is 4x7.
    # The inner W maps to the 4 entries of the 2x2 block C_block.
    # C_block 00 is at global indices 0,1,4,5.
    # C_block 01 is at 2,3,6,7.
    # C_block 10 is at 8,9,12,13.
    # C_block 11 is at 10,11,14,15.
    
    # Outer W contributions:
    # M1 -> C00 (+), C11 (+)
    # M2 -> C10 (+), C11 (-)
    # M3 -> C01 (+), C11 (+)
    # M4 -> C00 (+), C10 (+)
    # M5 -> C00 (-), C01 (+)
    # M6 -> C11 (+)
    # M7 -> C00 (+)
    
    outer_W_map = {
        0: {0: 1, 3: 1}, # M1: C00, C11
        1: {2: 1, 3: -1}, # M2: C10, C11
        2: {1: 1, 3: 1}, # M3: C01, C11
        3: {0: 1, 2: 1}, # M4: C00, C10
        4: {0: -1, 1: 1}, # M5: C00, C01
        5: {3: 1}, # M6: C11
        6: {0: 1} # M7: C00
    }
    
    U_global = np.zeros((16, 98))
    V_global = np.zeros((16, 98))
    W_global = np.zeros((32, 98)) # p*n = 8*4 = 32
    
    col = 0
    for k in range(7):
        # Get A and B block coefficients
        A_coeff = outer_coeffs[k][0]
        B_coeff = outer_coeffs[k][1]
        
        # Construct local U and V for the inner Strassen
        # Inner U is 4x7. We need to expand it to 16x7.
        U_local = np.zeros((16, 7))
        V_local = np.zeros((16, 7))
        
        # Map inner indices (0..3) to global indices based on blocks
        # Inner index 0 -> Block 00 (0,1,4,5)
        # Inner index 1 -> Block 01 (2,3,6,7)
        # Inner index 2 -> Block 10 (8,9,12,13)
        # Inner index 3 -> Block 11 (10,11,14,15)
        
        for inner_idx in range(4):
            block_id = inner_idx
            global_indices = block_indices_A[block_id]
            coeff = A_coeff.get(block_id, 0)
            if coeff != 0:
                # The inner U vector has values at 0..3
                # We add coeff * U_inner[:, l] to U_global at these indices
                for l in range(7):
                    val = U_inner[inner_idx, l]
                    if val != 0:
                        for g_idx in global_indices:
                            U_global[g_idx, col + l] += coeff * val
        
        for inner_idx in range(4):
            block_id = inner_idx
            global_indices = block_indices_B[block_id]
            coeff = B_coeff.get(block_id, 0)
            if coeff != 0:
                for l in range(7):
                    val = V_inner[inner_idx, l]
                    if val != 0:
                        for g_idx in global_indices:
                            V_global[g_idx, col + l] += coeff * val
        
        # Now W
        # Inner W is 4x7.
        # Inner index 0 -> C Block 00
        # Inner index 1 -> C Block 01
        # Inner index 2 -> C Block 10
        # Inner index 3 -> C Block 11
        
        # Outer W map tells which C blocks this M_k contributes to
        w_map = outer_W_map[k]
        
        for inner_idx in range(4):
            if inner_idx in w_map:
                c_block_coeff = w_map[inner_idx]
                c_block_id = inner_idx
                # C block indices
                if c_block_id == 0:
                    c_indices = [0, 1, 4, 5]
                elif c_block_id == 1:
                    c_indices = [2, 3, 6, 7]
                elif c_block_id == 2:
                    c_indices = [8, 9, 12, 13]
                else:
                    c_indices = [10, 11, 14, 15]
                
                # Note: W index in global is k*n + i?
                # Task says W[k*n+i, r].
                # k is the column index of C (0..7).
                # i is the row index of C (0..3).
                # C is 4x8.
                # C block 00: i=0,1; k=0,1.
                # Indices:
                # i=0, k=0 -> 0*4+0 = 0
                # i=0, k=1 -> 0*4+1 = 1
                # i=1, k=0 -> 1*4+0 = 4
                # i=1, k=1 -> 1*4+1 = 5
                # This matches [0,1,4,5].
                
                for l in range(7):
                    val = W_inner[inner_idx, l]
                    if val != 0:
                        for c_idx in c_indices:
                            W_global[c_idx, col + l] += c_block_coeff * val
        
        col += 7
    
    # Now we have the decomposition for the first 4 columns of B (k=0..3)
    # We need to do the same for the second 4 columns of B (k=4..7)
    # The A part is the same.
    # The B part uses blocks B00..B11 but shifted to columns 4..7.
    # The C part uses blocks C00..C11 but shifted to columns 4..7.
    
    # B blocks for second half:
    # B00 (second half) corresponds to j=0..1, k=4..5.
    # Indices:
    # j=0, k=4 -> 0*8+4 = 4? No.
    # B is 4x8.
    # B[j, k].
    # j=0..3, k=0..7.
    # Block B00 (second half): j=0,1; k=4,5.
    # Indices:
    # j=0, k=4 -> 0*8+4 = 4
    # j=0, k=5 -> 0*8+5 = 5
    # j=1, k=4 -> 1*8+4 = 12
    # j=1, k=5 -> 1*8+5 = 13
    # So indices [4, 5, 12, 13].
    
    block_indices_B2 = {
        0: [4, 5, 12, 13],
        1: [6, 7, 14, 15],
        2: [20, 21, 28, 29],
        3: [22, 23, 30, 31]
    }
    
    # C blocks for second half:
    # C00 (second half): i=0,1; k=4,5.
    # Indices:
    # i=0, k=4 -> 0*4+4 = 4
    # i=0, k=5 -> 0*4+5 = 5
    # i=1, k=4 -> 1*4+4 = 8
    # i=1, k=5 -> 1*4+5 = 9
    # So indices [4, 5, 8, 9].
    
    c_indices_2 = {
        0: [4, 5, 8, 9],
        1: [6, 7, 10, 11],
        2: [12, 13, 16, 17],
        3: [14, 15, 18, 19]
    }
    
    col = 49
    for k in range(7):
        A_coeff = outer_coeffs[k][0]
        B_coeff = outer_coeffs[k][1]
        
        U_local = np.zeros((16, 7))
        V_local = np.zeros((16, 7))
        
        for inner_idx in range(4):
            block_id = inner_idx
            global_indices = block_indices_A[block_id]
            coeff = A_coeff.get(block_id, 0)
            if coeff != 0:
                for l in range(7):
                    val = U_inner[inner_idx, l]
                    if val != 0:
                        for g_idx in global_indices:
                            U_global[g_idx, col + l] += coeff * val
        
        for inner_idx in range(4):
            block_id = inner_idx
            global_indices = block_indices_B2[block_id]
            coeff = B_coeff.get(block_id, 0)
            if coeff != 0:
                for l in range(7):
                    val = V_inner[inner_idx, l]
                    if val != 0:
                        for g_idx in global_indices:
                            V_global[g_idx, col + l] += coeff * val
        
        w_map = outer_W_map[k]
        
        for inner_idx in range(4):
            if inner_idx in w_map:
                c_block_coeff = w_map[inner_idx]
                c_block_id = inner_idx
                c_indices = c_indices_2[c_block_id]
                
                for l in range(7):
                    val = W_inner[inner_idx, l]
                    if val != 0:
                        for c_idx in c_indices:
                            W_global[c_idx, col + l] += c_block_coeff * val
        
        col += 7
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
