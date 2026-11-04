def construct():
    # 4x4 x 4x5 matrix multiplication
    # Part 1: 4x4 x 4x4 (columns 0-3 of B) using Strassen: rank 49
    # Part 2: 4x4 x 1 (column 4 of B) using rank-4 decomposition
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Get 2x2x2 Strassen base
    U2, V2, W2 = construct_2x2x2_strassen()
    
    # Outer Strassen terms for 4x4 = 2x(2x2) blocks
    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 -= M1, C01 += M1, C10 -= M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
    # M3 = A00(B01-B11) -> C00 += M3, C01 += M3
    # M4 = A11(B10-B00) -> C10 -= M4, C11 += M4
    # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 += M6, C00 -= M6
    # M7 = (A01-A11)(B10+B11) -> C01 -= M7, C10 += M7
    
    # For each outer term, define which A-blocks and B-blocks are involved and their coefficients
    # Then for each inner Strassen term, distribute the coefficients
    
    outer_defs = [
        # (a_blocks: {block_idx: coeff}, b_blocks: {block_idx: coeff}, c_blocks: {block_idx: coeff})
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: -1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),                        # M2
        ({0: 1}, {1: 1, 3: -1}, {0: 1, 1: 1}),                       # M3
        ({3: 1}, {2: 1, 0: -1}, {2: -1, 3: 1}),                      # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),                        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 0: -1}),                # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {1: -1, 2: 1}),                # M7
    ]
    
    for a_blocks, b_blocks, c_blocks in outer_defs:
        for r_i in range(7):
            U_new = [0] * 16  # 4x4
            V_new = [0] * 20  # 4x5 (only cols 0-3 used)
            W_new = [0] * 20  # 5x4
            
            for (ba, ca) in a_blocks.items():
                for (bb, cb) in b_blocks.items():
                    for (bc, cc) in c_blocks.items():
                        # A block ba is at row-block ba//2, col-block ba%2
                        # B block bb is at row-block bb//2, col-block bb%2
                        # C block bc is at row-block bc//2, col-block bc%2
                        # 
                        # For the inner 2x2x2 Strassen:
                        # U2[i2*2+j2, r_i] where i2=row within block, j2=col within block
                        # V2[j2*2+k2, r_i] where j2=row within block, k2=col within block  
                        # W2[k2*2+i2, r_i] where k2=row within block, i2=col within block
                        #
                        # Global indices:
                        # A: i_global = 2*(ba//2) + i2, j_global = 2*(ba%2) + j2
                        # B: j_global = 2*(bb//2) + j2, k_global = 2*(bb%2) + k2
                        # C: k_global = 2*(bc//2) + k2, i_global = 2*(bc%2) + i2
                        
                        for i2 in range(2):
                            for j2 in range(2):
                                u2 = U2[i2*2 + j2][r_i]
                                if u2 == 0:
                                    continue
                                i_g = 2*(ba//2) + i2
                                j_g = 2*(ba%2) + j2
                                coeff_u = ca * u2
                                if coeff_u != 0:
                                    U_new[i_g*4 + j_g] += coeff_u
                        
                        for j2 in range(2):
                            for k2 in range(2):
                                v2 = V2[j2*2 + k2][r_i]
                                if v2 == 0:
                                    continue
                                j_g = 2*(bb//2) + j2
                                k_g = 2*(bb%2) + k2
                                coeff_v = cb * v2
                                if coeff_v != 0:
                                    V_new[j_g*5 + k_g] += coeff_v
                        
                        for k2 in range(2):
                            for i2 in range(2):
                                w2 = W2[k2*2 + i2][r_i]
                                if w2 == 0:
                                    continue
                                k_g = 2*(bc//2) + k2
                                i_g = 2*(bc%2) + i2
                                coeff_w = cc * w2
                                if coeff_w != 0:
                                    W_new[k_g*4 + i_g] += coeff_w
            
            U_terms.append(U_new)
            V_terms.append(V_new)
            W_terms.append(W_new)
    
    # Part 2: 4x4 x 1 (column 4 of B), rank 4
    for r in range(4):
        U_new = [0] * 16
        V_new = [0] * 20
        W_new = [0] * 20
        for i in range(4):
            U_new[i*4 + r] = 1
        V_new[r*5 + 4] = 1
        for i in range(4):
            W_new[16 + i] = 1
        U_terms.append(U_new)
        V_terms.append(V_new)
        W_terms.append(W_new)
    
    return {"U": U_terms, "V": V_terms, "W": W_terms}
