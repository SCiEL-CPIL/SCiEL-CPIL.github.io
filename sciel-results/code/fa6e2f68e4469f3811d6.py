"""Program for matrix multiplication 4x4x9 using Strassen-based decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 114: 49 (4x4x4) + 49 (4x4x4) + 16 (4x4x1)
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, -1, 0, -1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 1, 0, 0]
    ]
    
    # Helper to add a 4x4x4 Strassen block
    # offset: starting index in U, V, W
    # b_col_offset: starting column in B (0, 4)
    # c_col_offset: starting column in C (0, 4)
    def add_strassen_4x4x4(offset, b_col_offset, c_col_offset):
        # The 7 Strassen macro-steps
        # Each step is a 2x2x2 multiplication
        # We need to define the linear combinations for A and B blocks
        
        # A blocks: A00, A01, A10, A11
        # Indices in global U (0..15):
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        # B blocks: B00, B01, B10, B11
        # Indices in global V (0..35)
        # B00: b_col_offset + 0,1,4,5
        # B01: b_col_offset + 2,3,6,7
        # B10: b_col_offset + 8,9,12,13
        # B11: b_col_offset + 10,11,14,15
        
        # C blocks: C00, C01, C10, C11
        # Indices in global W (0..35)
        # C00: c_col_offset + 0,1,4,5
        # C01: c_col_offset + 2,3,6,7
        # C10: c_col_offset + 8,9,12,13
        # C11: c_col_offset + 10,11,14,15
        
        # Define the 7 steps
        # Each step has:
        # A_comb: list of 4 coeffs for A00, A01, A10, A11
        # B_comb: list of 4 coeffs for B00, B01, B10, B11
        # C_comb: list of 4 coeffs for C00, C01, C10, C11 (from W2)
        
        steps = [
            # M1 = (A00+A11)(B00+B11)
            {'A': [1, 0, 0, 1], 'B': [1, 0, 0, 1]},
            # M2 = (A10+A11)B00
            {'A': [0, 0, 1, 1], 'B': [1, 0, 0, 0]},
            # M3 = A00(B01-B11)
            {'A': [1, 0, 0, 0], 'B': [0, 1, 0, -1]},
            # M4 = A11(B10-B00)
            {'A': [0, 0, 0, 1], 'B': [-1, 0, 1, 0]},
            # M5 = (A00+A01)B11
            {'A': [1, 1, 0, 0], 'B': [0, 0, 0, 1]},
            # M6 = (A10-A00)(B00+B01)
            {'A': [-1, 0, 1, 0], 'B': [1, 1, 0, 0]},
            # M7 = (A01-A11)(B10+B11)
            {'A': [0, 1, 0, -1], 'B': [0, 0, 1, 1]}
        ]
        
        # C_comb comes from W2 rows
        # W2[0] -> C00, W2[1] -> C01, W2[2] -> C10, W2[3] -> C11
        # But W2 is for the 2x2x2 output.
        # For each macro-step, the output is a 2x2 matrix M_k.
        # We add M_k to the final C blocks with specific weights?
        # No, the Strassen algorithm computes C directly from M1..M7.
        # C00 = M1 - M4 + M7 - M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # So for each macro-step k, we compute the 2x2x2 factors.
        # The U factors for step k are: U2 * A_comb (broadcast)
        # The V factors for step k are: V2 * B_comb (broadcast)
        # The W factors for step k are: W2 * C_weights_k
        
        # C_weights for each step:
        # M1: C00: 1, C01: 0, C10: 0, C11: 1
        # M2: C00: 0, C01: 0, C10: 1, C11: -1
        # M3: C00: 0, C01: 1, C10: 0, C11: 1
        # M4: C00: -1, C01: 0, C10: 1, C11: 0
        # M5: C00: 0, C01: 1, C10: 0, C11: 1
        # M6: C00: -1, C01: 0, C10: 0, C11: 0
        # M7: C00: 1, C01: 0, C10: 0, C11: 0
        
        c_weights = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1], # M2
            [0, 1, 0, 1], # M3
            [-1, 0, 1, 0], # M4
            [0, 1, 0, 1], # M5
            [-1, 0, 0, 0], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        for k in range(7):
            a_comb = steps[k]['A']
            b_comb = steps[k]['B']
            cw = c_weights[k]
            
            # Iterate over the 7 scalar multiplications in the 2x2x2 block
            for r in range(7):
                current_rank = offset + k * 7 + r
                
                # Fill U
                # U2[i][r] is coeff for local A element i
                # Local A elements are linear combos of global A blocks
                # U2[0] corresponds to A00, U2[1] to A01, etc?
                # No, U2 rows are A00, A01, A10, A11 of the 2x2 matrix.
                # So for each row i in U2 (0..3):
                #   coeff = U2[i][r]
                #   This applies to the i-th block of the combined A.
                #   The i-th block is a 2x2 matrix.
                #   We need to map this to global U.
                
                # This logic is tricky.
                # Let's just iterate over the 16 elements of A.
                # For each global U index (0..15):
                #   Determine which block (0..3) and local pos (0..3) it belongs to.
                #   Block 0 (A00): indices 0,1,4,5
                #   Block 1 (A01): indices 2,3,6,7
                #   Block 2 (A10): indices 8,9,12,13
                #   Block 3 (A11): indices 10,11,14,15
                
                # We need the value of U for this global index.
                # It is sum over local i of (U2[i][r] * a_comb[i] * delta(local_pos))
                # Actually, a_comb[i] scales the entire block i.
                # So for global index g:
                #   block = g // 4 ? No.
                #   A is 4x4.
                #   A00 is rows 0,1 cols 0,1.
                #   Indices:
                #   0: (0,0) -> Block 0, pos 0
                #   1: (0,1) -> Block 0, pos 1
                #   2: (0,2) -> Block 1, pos 0
                #   3: (0,3) -> Block 1, pos 1
                #   4: (1,0) -> Block 0, pos 2
                #   5: (1,1) -> Block 0, pos 3
                #   6: (1,2) -> Block 1, pos 2
                #   7: (1,3) -> Block 1, pos 3
                #   8: (2,0) -> Block 2, pos 0
                #   9: (2,1) -> Block 2, pos 1
                #   10: (2,2) -> Block 3, pos 0
                #   11: (2,3) -> Block 3, pos 1
                #   12: (3,0) -> Block 2, pos 2
                #   13: (3,1) -> Block 2, pos 3
                #   14: (3,2) -> Block 3, pos 2
                #   15: (3,3) -> Block 3, pos 3
                
                # U2 rows:
                # 0: A00
                # 1: A01
                # 2: A10
                # 3: A11
                
                # So for global index g:
                #   block = ...
                #   local_pos = ...
                #   U_val = U2[block][r] * a_comb[block]
                
                for g in range(16):
                    row = g // 4
                    col = g % 4
                    block = (row // 2) * 2 + (col // 2)
                    # U2 index is block
                    val = U2[block][r] * a_comb[block]
                    U[g][current_rank] += val
                    
                # Fill V
                # Similar logic for B
                # B00: b_col_offset + 0,1,4,5
                # B01: b_col_offset + 2,3,6,7
                # B10: b_col_offset + 8,9,12,13
                # B11: b_col_offset + 10,11,14,15
                
                for g in range(16):
                    row = g // 4
                    col = g % 4
                    block = (row // 2) * 2 + (col // 2)
                    global_idx = b_col_offset + g
                    val = V2[block][r] * b_comb[block]
                    V[global_idx][current_rank] += val
                    
                # Fill W
                # W2 rows:
                # 0: C00
                # 1: C01
                # 2: C10
                # 3: C11
                
                # C_weights[k] gives weights for C00, C01, C10, C11
                # So for each local c_block (0..3):
                #   W_val = W2[c_block][r] * cw[c_block]
                
                for g in range(16):
                    row = g // 4
                    col = g % 4
                    block = (row // 2) * 2 + (col // 2)
                    global_idx = c_col_offset + g
                    val = W2[block][r] * cw[block]
                    W[global_idx][current_rank] += val

    # Add Block 1: 4x4x4
    add_strassen_4x4x4(0, 0, 0)
    
    # Add Block 2: 4x4x4
    add_strassen_4x4x4(49, 4, 4)
    
    # Add Block 3: 4x4x1 (Schoolbook)
    offset = 98
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 9 + 8 # B col 8
            w_idx = 8 * 4 + i # W[k*n+i], k=8, i=i
            U[u_idx][offset] = 1
            V[v_idx][offset] = 1
            W[w_idx][offset] = 1
            offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
