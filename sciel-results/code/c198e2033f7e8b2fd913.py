import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A is 2x4, B is 4x6.
    # Split B into three 2x2 blocks horizontally: B = [B1 | B2 | B3]
    # A is split into two 2x2 blocks horizontally: A = [A1 | A2]
    # This doesn't align for direct multiplication.
    # Correct split:
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2.
    # C = A1 * [B11 B12 B13] + A2 * [B21 B22 B23]
    # C = [A1*B11 + A2*B21 | A1*B12 + A2*B22 | A1*B13 + A2*B23]
    # Each term like A1*B11 + A2*B21 is a sum of two 2x2x2 multiplications.
    # Rank of 2x2x2 is 7 (Strassen).
    # Sum of two independent 2x2x2 multiplications has rank 14.
    # There are 3 such terms in C.
    # Total rank = 3 * 14 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to map this to the global indices.
    # Global A indices: i (0,1), j (0..3).
    # A1 corresponds to j=0,1. A2 corresponds to j=2,3.
    # Global B indices: j (0..3), k (0..5).
    # B11: j=0,1 k=0,1. B12: j=0,1 k=2,3. B13: j=0,1 k=4,5.
    # B21: j=2,3 k=0,1. B22: j=2,3 k=2,3. B23: j=2,3 k=4,5.
    # Global C indices: k (0..5), i (0,1).
    # C1 (k=0,1): A1*B11 + A2*B21.
    # C2 (k=2,3): A1*B12 + A2*B22.
    # C3 (k=4,5): A1*B13 + A2*B23.
    
    col = 0
    for block_idx in range(3):
        # Block 0: k=0,1. B11 (j=0,1), B21 (j=2,3).
        # Block 1: k=2,3. B12 (j=0,1), B22 (j=2,3).
        # Block 2: k=4,5. B13 (j=0,1), B23 (j=2,3).
        
        # Term 1: A1 * B_top
        # A1 is 2x2 (i=0,1; j=0,1).
        # B_top is 2x2 (j=0,1; k=2*block_idx .. 2*block_idx+1).
        
        # Term 2: A2 * B_bot
        # A2 is 2x2 (i=0,1; j=2,3).
        # B_bot is 2x2 (j=2,3; k=2*block_idx .. 2*block_idx+1).
        
        # We add 14 rank terms for this block.
        # 7 for A1*B_top, 7 for A2*B_bot.
        
        # Strassen terms for A1*B_top
        # A1 indices: i in 0,1; j in 0,1.
        # B_top indices: j in 0,1; k in 2*block_idx, 2*block_idx+1.
        
        # Strassen terms for A2*B_bot
        # A2 indices: i in 0,1; j in 2,3.
        # B_bot indices: j in 2,3; k in 2*block_idx, 2*block_idx+1.
        
        # We need to define the Strassen coefficients.
        # For a 2x2x2 multiplication X*Y:
        # U, V, W tensors.
        
        def add_strassen_block(a_i_base, a_j_base, b_j_base, b_k_base, c_k_base, c_i_base, start_col):
            # a_i_base, a_j_base: base indices for A (2x2)
            # b_j_base, b_k_base: base indices for B (2x2)
            # c_k_base, c_i_base: base indices for C (2x2)
            
            # Strassen 7 terms
            # 1. (A00+A11)(B00+B11)
            # 2. (A10+A11)B00
            # 3. A00(B10-B11)
            # 4. A11(B01-B00)
            # 5. (A00+A01)B11
            # 6. (A10-A00)(B00+B01)
            # 7. (A01-A11)(B10+B11)
            
            # Coefficients for U, V, W
            # U (A part), V (B part), W (C part)
            
            # Term 1
            U_vals = [(0,1), (1,1)] # A00, A11
            V_vals = [(0,0), (1,1)] # B00, B11
            W_val = [(0,0), (0,1), (1,0), (1,1)] # C00, C01, C10, C11 with signs?
            # C00 += 1, C01 += 1, C10 += 1, C11 += 1? No.
            # C00 = M1+M4-M5+M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3+M6
            
            # We will just iterate through the 7 terms and add to the global matrices.
            
            terms = [
                # (U_coeff, V_coeff, W_coeff)
                # U is 2x2, V is 2x2, W is 2x2
                # Term 1: U=(1,1,1,1), V=(1,1,1,1), W=(1,1,1,1)
                ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]),
                # Term 2: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,0,0; 1,1,0,0) -> C10, C11?
                # C10 += M2, C11 += -M2
                ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
                # Term 3: U=(1,0,0,0), V=(0,1,-1), W=(0,1,0,0; 0,0,1,0) -> C01, C10?
                # C01 += M3, C11 += M3
                ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
                # Term 4: U=(1,1,0,0), V=(0,1,-1), W=(1,0,0,0; 0,1,0,0) -> C00, C10
                # C00 += M4, C10 += M4
                ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
                # Term 5: U=(1,1,0,0), V=(0,0,1,1), W=(0,0,0,0; 0,1,-1,0) -> C01, C00?
                # C01 += M5, C00 -= M5
                ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)]),
                # Term 6: U=(0,0,1,1), V=(1,1,1,1), W=(0,0,0,0; 0,0,0,1) -> C11
                # C11 += M6
                ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
                # Term 7: U=(0,0,1,1), V=(1,1,1,1), W=(0,0,1,0; 0,1,0,0) -> C00, C11?
                # C00 += M7, C11 += M7
                ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ]
            
            for u_terms, v_terms, w_terms in terms:
                for (ui, uj, u_val) in u_terms:
                    for (vi, vj, v_val) in v_terms:
                        for (wi, wj, w_val) in w_terms:
                            # Global indices
                            # A: i = a_i_base + ui, j = a_j_base + uj
                            # B: j = b_j_base + vi, k = b_k_base + vj
                            # C: k = c_k_base + wi, i = c_i_base + wj
                            
                            a_idx = (a_i_base + ui) * m + (a_j_base + uj)
                            b_idx = (b_j_base + vi) * p + (b_k_base + vj)
                            c_idx = (c_k_base + wi) * n + (c_i_base + wj)
                            
                            U[a_idx][start_col] += u_val * v_val * w_val
                            
                start_col += 1
            return start_col

        # Block 0
        col = add_strassen_block(0, 0, 0, 0, 0, 0, col) # A1, B11
        col = add_strassen_block(0, 2, 2, 0, 0, 0, col) # A2, B21
        
        # Block 1
        col = add_strassen_block(0, 0, 0, 2, 2, 0, col) # A1, B12
        col = add_strassen_block(0, 2, 2, 2, 2, 0, col) # A2, B22
        
        # Block 2
        col = add_strassen_block(0, 0, 0, 4, 4, 0, col) # A1, B13
        col = add_strassen_block(0, 2, 2, 4, 4, 0, 0, col) # A2, B23
        
    return {"U": U, "V": V, "W": W}
