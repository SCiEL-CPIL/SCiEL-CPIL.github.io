<<<<<<< SEARCH
    # --- Part 2: 4x4x3 using Strassen(7) + Schoolbook(4) per (2x2)*(2x3) block ---
    # A is 4x4, B is 4x3, C is 4x3.
    # Split into 2x2 blocks of A and 2x3 blocks of B.
    # For each (2x2)*(2x3) block pair, use rank-44 decomposition:
    #   - Strassen (rank 7) for the 2x2x2 part (cols 0-1 of B block)
    #   - Schoolbook (rank 4) for the 2x2x1 part (col 2 of B block)
    # Total: 4 block pairs * 11 = 44
    
    offset = rank_444
    col = 0
    
    # Strassen definitions for 2x2x2 (same as before)
    strassen_222 = [
        # (X_def, Y_def, Dest)
        # X_def: list of (block_row, block_col, coeff) for A (2x2)
        # Y_def: list of (block_row, block_col, coeff) for B (2x2)
        # Dest: list of (block_row, block_col, coeff) for C (2x2)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # For each 2x2 block of A and 2x3 block of B
    for ai in range(2):  # A block row (0 or 1)
        for aj in range(2):  # A block col (0 or 1)
            for bi in range(2):  # B block row (0 or 1)
                for bk in range(2):  # B block col (0 or 1)
                    # A block: rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1
                    # B block: rows 2*bi to 2*bi+1, cols 4+2*bk to 4+2*bk+2 (local cols 0-2)
                    # C block: rows 2*ai to 2*ai+1, cols 4+2*bk to 4+2*bk+2 (local cols 0-2)
                    
                    # Strassen part: handles local cols 0-1 of B block
                    for s_idx, (X_def, Y_def, Dest) in enumerate(strassen_222):
                        r = offset + col
                        
                        # Build U column
                        U_col = np.zeros(nm)
                        for (bx, by, cx) in X_def:
                            for (i_prime, j_prime, cu) in X_def:
                                i = 2 * ai + 2 * bx + i_prime
                                j = 2 * aj + 2 * by + j_prime
                                a_idx = i * m + j
                                U_col[a_idx] += cx * cu
                        
                        U[:, r] = U_col
                        
                        # Build V column
                        V_col = np.zeros(mp)
                        for (bx, by, cy) in Y_def:
                            for (i_prime, j_prime, cv) in Y_def:
                                row_b = 2 * bi + 2 * bx + i_prime
                                col_b = 4 + 2 * bk + 2 * by + j_prime
                                b_idx = row_b * p + col_b
                                V_col[b_idx] += cy * cv
                        
                        V[:, r] = V_col
                        
                        # Build W column
                        W_col = np.zeros(pn)
                        for (dx, dy, cd) in Dest:
                            for (i_prime, j_prime, co) in Dest:
                                row_c = 2 * ai + 2 * dx + i_prime
                                col_c = 4 + 2 * bk + 2 * dy + j_prime
                                c_idx = col_c * n + row_c
                                W_col[c_idx] += cd * co
                        
                        W[:, r] = W_col
                        col += 1
                    
                    # Schoolbook part: handles local col 2 of B block
                    # C[i, 2] = sum_j A[i, j] B[j, 2]
                    for i_prime in range(2):
                        for j_prime in range(2):
                            r = offset + col
                            i = 2 * ai + i_prime
                            j = 2 * aj + j_prime
                            a_idx = i * m + j
                            b_idx = j * p + (4 + 2 * bk + 2)
                            c_idx = (4 + 2 * bk + 2) * n + i
                            
                            U[a_idx, r] = 1.0
                            V[b_idx, r] = 1.0
                            W[c_idx, r] = 1.0
                            col += 1
                            
=======
    # --- Part 2: 4x4x3 using Strassen(7) + Schoolbook(4) per (2x2)*(2x3) block ---
    # A is 4x4, B is 4x3, C is 4x3.
    # Split into 2x2 blocks of A and 2x3 blocks of B.
    # For each (2x2)*(2x3) block pair, use rank-44 decomposition:
    #   - Strassen (rank 7) for the 2x2x2 part (cols 0-1 of B block)
    #   - Schoolbook (rank 4) for the 2x2x1 part (col 2 of B block)
    # Total: 4 block pairs * 11 = 44
    
    offset = rank_444
    col = 0
    
    # Strassen definitions for 2x2x2: (u_def, v_def, out_def)
    # u_def: linear form of X (2x2), list of (row, col, coeff)
    # v_def: linear form of Y (2x2), list of (row, col, coeff)
    # out_def: linear form of Z (2x2), list of (row, col, coeff)
    strassen_222 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # For each 2x2 block of A and 2x3 block of B
    for ai in range(2):  # A block row (0 or 1)
        for aj in range(2):  # A block col (0 or 1)
            for bi in range(2):  # B block row (0 or 1)
                for bk in range(2):  # B block col (0 or 1)
                    # A block: rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1
                    # B block: rows 2*bi to 2*bi+1, cols 4+2*bk to 4+2*bk+2 (local cols 0-2)
                    # C block: rows 2*ai to 2*ai+1, cols 4+2*bk to 4+2*bk+2 (local cols 0-2)
                    
                    # Strassen part: handles local cols 0-1 of B block
                    for s_idx, (u_def, v_def, out_def) in enumerate(strassen_222):
                        r = offset + col
                        
                        # Build U column
                        # U[a, r] = sum over u_def of coeff_u * A[2*ai + i', 2*aj + j']
                        U_col = np.zeros(nm)
                        for (i_prime, j_prime, cu) in u_def:
                            i = 2 * ai + i_prime
                            j = 2 * aj + j_prime
                            a_idx = i * m + j
                            U_col[a_idx] += cu
                        
                        U[:, r] = U_col
                        
                        # Build V column
                        # V[b, r] = sum over v_def of coeff_v * B[2*bi + i', 4+2*bk + j']
                        V_col = np.zeros(mp)
                        for (i_prime, j_prime, cv) in v_def:
                            row_b = 2 * bi + i_prime
                            col_b = 4 + 2 * bk + j_prime
                            b_idx = row_b * p + col_b
                            V_col[b_idx] += cv
                        
                        V[:, r] = V_col
                        
                        # Build W column
                        # W[c, r] = sum over out_def of coeff_out * C[2*ai + i', 4+2*bk + j']
                        W_col = np.zeros(pn)
                        for (i_prime, j_prime, co) in out_def:
                            row_c = 2 * ai + i_prime
                            col_c = 4 + 2 * bk + j_prime
                            c_idx = col_c * n + row_c
                            W_col[c_idx] += co
                        
                        W[:, r] = W_col
                        col += 1
                    
                    # Schoolbook part: handles local col 2 of B block
                    # C[i, 2] = sum_j A[i, j] B[j, 2]
                    for i_prime in range(2):
                        for j_prime in range(2):
                            r = offset + col
                            i = 2 * ai + i_prime
                            j = 2 * aj + j_prime
                            a_idx = i * m + j
                            b_idx = j * p + (4 + 2 * bk + 2)
                            c_idx = (4 + 2 * bk + 2) * n + i
                            
                            U[a_idx, r] = 1.0
                            V[b_idx, r] = 1.0
                            W[c_idx, r] = 1.0
                            col += 1
                            
>>>>>>> REPLACE

