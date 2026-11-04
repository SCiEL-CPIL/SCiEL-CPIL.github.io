    def strassen_factors(n):
        if n == 1:
            return [[1]], [[1]], [[1]]
        if n == 2:
            # Hardcoded 2x2x2
            U = [
                [1, 0, 0, 0], # M1
                [0, 1, 0, 0], # M2
                [0, 0, 1, 0], # M3
                [0, 0, 0, 1]  # M4 ... wait, U is 4x7
            ]
            # Let's define U, V, W as lists of columns (rank-1 tensors)
            # Col 0: M1
            u0 = [1, 0, 0, 1]
            v0 = [1, 0, 0, 1]
            w0 = [1, 0, 0, 1] # C11 gets M1, C22 gets M1
            # Wait, W is the output coefficients.
            # For M1, it contributes to C11 (1), C22 (1).
            # So w0 = [1, 0, 0, 1] is correct for C indices 0,1,2,3 (C11, C12, C21, C22).
            
            # Col 1: M2
            u1 = [0, 0, 1, 1]
            v1 = [1, 0, 0, 0]
            w1 = [0, 0, 1, -1] # C21 gets M2, C22 gets -M2
            
            # Col 2: M3
            u2 = [1, 0, 0, 0]
            v2 = [0, 0, 1, -1]
            w2 = [0, 1, 0, 1] # C12 gets M3, C22 gets M3
            
            # Col 3: M4
            u3 = [0, 0, 0, 1]
            v3 = [0, 1, 1, 0]
            w3 = [1, 0, 1, 0] # C11 gets M4, C21 gets M4
            
            # Col 4: M5
            u4 = [1, 1, 0, 0]
            v4 = [0, 0, 0, 1]
            w4 = [-1, 1, 0, 0] # C11 gets -M5, C12 gets M5
            
            # Col 5: M6
            u5 = [-1, 0, 1, 0]
            v5 = [1, 1, 0, 0]
            w5 = [0, 0, 0, 1] # C22 gets M6
            
            # Col 6: M7
            u6 = [0, 1, 0, -1]
            v6 = [0, 0, 1, 1]
            w6 = [1, 0, 0, 0] # C11 gets M7
            
            U = [u0, u1, u2, u3, u4, u5, u6]
            V = [v0, v1, v2, v3, v4, v5, v6]
            W = [w0, w1, w2, w3, w4, w5, w6]
            # Convert to row-major matrices
            U_mat = np.array(U).T
            V_mat = np.array(V).T
            W_mat = np.array(W).T
            return U_mat, V_mat, W_mat

        # Recursive case
        m = n // 2
        U_m, V_m, W_m = strassen_factors(m)
        rank_m = U_m.shape[1]
        rank_n = 7 * rank_m
        
        U_n = np.zeros((n*n, rank_n))
        V_n = np.zeros((n*n, rank_n))
        W_n = np.zeros((n*n, rank_n))
        
        # Strassen 2x2x2 structure
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # We need to map the blocks of A (size m x m) to the vector of length n*n.
        # A = [[A00, A01], [A10, A11]]
        # Indices:
        # A00: 0..m-1 rows, 0..m-1 cols -> 0..m*m-1
        # A01: 0..m-1 rows, m..n-1 cols -> m*m..2*m*m-1
        # A10: m..n-1 rows, 0..m-1 cols -> 2*m*m..3*m*m-1
        # A11: m..n-1 rows, m..n-1 cols -> 3*m*m..4*m*m-1
        
        def get_block_indices(matrix_size, block_r, block_c):
            start_r = block_r * m
            start_c = block_c * m
            indices = []
            for r in range(m):
                for c in range(m):
                    indices.append((start_r + r) * matrix_size + (start_c + c))
            return indices
            
        idx_A00 = get_block_indices(n, 0, 0)
        idx_A01 = get_block_indices(n, 0, 1)
        idx_A10 = get_block_indices(n, 1, 0)
        idx_A11 = get_block_indices(n, 1, 1)
        
        # Same for B
        idx_B00 = get_block_indices(n, 0, 0)
        idx_B01 = get_block_indices(n, 0, 1)
        idx_B10 = get_block_indices(n, 1, 0)
        idx_B11 = get_block_indices(n, 1, 1)
        
        # Same for C (Output)
        idx_C00 = get_block_indices(n, 0, 0)
        idx_C01 = get_block_indices(n, 0, 1)
        idx_C10 = get_block_indices(n, 1, 0)
        idx_C11 = get_block_indices(n, 1, 1)
        
        # The 7 outer steps
        outer_steps = [
            # (U_blocks, V_blocks, W_blocks, signs)
            # U_blocks: list of (block_name, sign)
            # V_blocks: list of (block_name, sign)
            # W_blocks: list of (block_name, sign)
            # This is tricky because the inner multiplication is m x m.
            # The inner U_m, V_m, W_m are for m x m.
            # So for each outer step, we take the linear combination of blocks,
            # which forms an m x m matrix, and multiply it using the inner factors.
            
            # Let's define the linear combinations for A and B for each of the 7 steps.
            # Step 1: A = A00 + A11, B = B00 + B11
            # Step 2: A = A10 + A11, B = B00
            # Step 3: A = A00, B = B10 - B11
            # Step 4: A = A11, B = B01 - B00
            # Step 5: A = A00 + A01, B = B11
            # Step 6: A = A10 - A00, B = B00 + B01
            # Step 7: A = A01 - A11, B = B10 + B11
        ]
        
        # For each step, we form the effective U_vec and V_vec for the m x m multiplication.
        # Then we combine with U_m, V_m, W_m.
        
        for step in range(7):
            # Determine coefficients for A blocks
            a_coeffs = {0: 0, 1: 0, 2: 0, 3: 0} # 0:00, 1:01, 2:10, 3:11
            b_coeffs = {0: 0, 1: 0, 2: 0, 3: 0}
            
            if step == 0:
                a_coeffs[0] = 1; a_coeffs[3] = 1
                b_coeffs[0] = 1; b_coeffs[3] = 1
            elif step == 1:
                a_coeffs[2] = 1; a_coeffs[3] = 1
                b_coeffs[0] = 1
            elif step == 2:
                a_coeffs[0] = 1
                b_coeffs[2] = 1; b_coeffs[3] = -1
            elif step == 3:
                a_coeffs[3] = 1
                b_coeffs[1] = 1; b_coeffs[0] = -1
            elif step == 4:
                a_coeffs[0] = 1; a_coeffs[1] = 1
                b_coeffs[3] = 1
            elif step == 5:
                a_coeffs[2] = 1; a_coeffs[0] = -1
                b_coeffs[0] = 1; b_coeffs[1] = 1
            elif step == 6:
                a_coeffs[1] = 1; a_coeffs[3] = -1
                b_coeffs[2] = 1; b_coeffs[3] = 1
            
            # Construct the linear map for A
            # U_effective is a vector of length m*m
            # It is the sum of a_coeffs[block] * A_block
            # We need to map this to the U_n matrix.
            # For each inner term t (0..rank_m-1):
            # The contribution to U_n is U_m[:, t] applied to the effective A.
            
            col_offset = step * rank_m
            
            for t in range(rank_m):
                # U_m[:, t] is a vector of length m*m
                # We need to distribute this to the n*n indices of A.
                u_m_t = U_m[:, t]
                
                # u_m_t[i] is the coefficient of A_eff[i]
                # A_eff is a linear combination of blocks.
                # A_eff[r, c] = sum_{block} a_coeffs[block] * A_block[r, c]
                # So U_n[idx_A_block[r,c], col] += a_coeffs[block] * u_m_t[r*m+c]
                
                for r in range(m):
                    for c in range(m):
                        coeff = u_m_t[r*m+c]
                        if coeff == 0: continue
                        # Add to A00
                        if a_coeffs[0] != 0:
                            U_n[idx_A00[r*m+c], col_offset+t] += a_coeffs[0] * coeff
                        if a_coeffs[1] != 0:
                            U_n[idx_A01[r*m+c], col_offset+t] += a_coeffs[1] * coeff
                        if a_coeffs[2] != 0:
                            U_n[idx_A10[r*m+c], col_offset+t] += a_coeffs[2] * coeff
                        if a_coeffs[3] != 0:
                            U_n[idx_A11[r*m+c], col_offset+t] += a_coeffs[3] * coeff
                
                # Same for V
                v_m_t = V_m[:, t]
                for r in range(m):
                    for c in range(m):
                        coeff = v_m_t[r*m+c]
                        if coeff == 0: continue
                        if b_coeffs[0] != 0:
                            V_n[idx_B00[r*m+c], col_offset+t] += b_coeffs[0] * coeff
                        if b_coeffs[1] != 0:
                            V_n[idx_B01[r*m+c], col_offset+t] += b_coeffs[1] * coeff
                        if b_coeffs[2] != 0:
                            V_n[idx_B10[r*m+c], col_offset+t] += b_coeffs[2] * coeff
                        if b_coeffs[3] != 0:
                            V_n[idx_B11[r*m+c], col_offset+t] += b_coeffs[3] * coeff
                
                # Same for W
                w_m_t = W_m[:, t]
                # The output of the inner multiplication is a m x m matrix M.
                # This M contributes to the blocks of C.
                # C00 gets M, C11 gets M, etc?
                # No, the outer Strassen steps determine which C blocks get the result.
                # Step 0 (M1): C00 += M, C11 += M, C01 -= M, C10 += M?
                # Let's check Strassen formulas again.
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # So for Step 0 (M1):
                # C00 += 1*M, C01 += 0, C10 += 0, C11 += 1*M
                # For Step 1 (M2):
                # C00 += 0, C01 += 0, C10 += 1*M, C11 += -1*M
                # For Step 2 (M3):
                # C00 += 0, C01 += 1*M, C10 += 0, C11 += 1*M
                # For Step 3 (M4):
                # C00 += 1*M, C01 += 0, C10 += 1*M, C11 += 0
                # For Step 4 (M5):
                # C00 += -1*M, C01 += 1*M, C10 += 0, C11 += 0
                # For Step 5 (M6):
                # C00 += 0, C01 += 0, C10 += 0, C11 += 1*M
                # For Step 6 (M7):
                # C00 += 1*M, C01 += 0, C10 += 0, C11 += 0
                
                w_coeffs = {0: 0, 1: 0, 2: 0, 3: 0}
                if step == 0:
                    w_coeffs[0] = 1; w_coeffs[3] = 1
                elif step == 1:
                    w_coeffs[2] = 1; w_coeffs[3] = -1
                elif step == 2:
                    w_coeffs[1] = 1; w_coeffs[3] = 1
                elif step == 3:
                    w_coeffs[0] = 1; w_coeffs[2] = 1
                elif step == 4:
                    w_coeffs[0] = -1; w_coeffs[1] = 1
                elif step == 5:
                    w_coeffs[3] = 1
                elif step == 6:
                    w_coeffs[0] = 1
                
                for r in range(m):
                    for c in range(m):
                        coeff = w_m_t[r*m+c]
                        if coeff == 0: continue
                        if w_coeffs[0] != 0:
                            W_n[idx_C00[r*m+c], col_offset+t] += w_coeffs[0] * coeff
                        if w_coeffs[1] != 0:
                            W_n[idx_C01[r*m+c], col_offset+t] += w_coeffs[1] * coeff
                        if w_coeffs[2] != 0:
                            W_n[idx_C10[r*m+c], col_offset+t] += w_coeffs[2] * coeff
                        if w_coeffs[3] != 0:
                            W_n[idx_C11[r*m+c], col_offset+t] += w_coeffs[3] * coeff
        
        return U_n, V_n, W_n
    