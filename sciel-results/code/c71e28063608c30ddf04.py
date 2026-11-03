"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # Keys are local indices 0..3 (00, 01, 10, 11)
    # U/V local index mapping: idx = row * 2 + col
    # W local index mapping: idx = row * 2 + col (but W is stored as W[k*n+i] in global, 
    # so in local block logic, we just map local (r,c) to global)
    
    # Strassen components for P*Q=R (2x2)
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block definitions:
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x4. Split into 2x2 blocks:
    #   B_LL (rows 0,1, cols 0,1)
    #   B_LR (rows 0,1, cols 2,3)
    #   B_RL (rows 2,3, cols 0,1)
    #   B_RR (rows 2,3, cols 2,3)
    # C is 2x4. Split into C_L (cols 0,1) and C_R (cols 2,3).
    
    # Multiplications:
    # C_L = A_L * B_LL + A_R * B_RL
    # C_R = A_L * B_LR + A_R * B_RR
    
    # Each term is (A_block, B_block, C_block)
    # A_block: 'L' (col offset 0) or 'R' (col offset 2)
    # B_block: 'LL' (r0=0,c0=0), 'LR' (r0=0,c0=2), 'RL' (r0=2,c0=0), 'RR' (r0=2,c0=2)
    # C_block: 'L' (col offset 0) or 'R' (col offset 2)
    
    blocks = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0), # A_L * B_LL -> C_L
        (2, 2, 0, 0), # A_R * B_RL -> C_L
        (0, 0, 2, 2), # A_L * B_LR -> C_R
        (2, 2, 2, 2), # A_R * B_RR -> C_R
    ]
    
    current_rank = 0
    for a_co, b_ro, b_co, c_co in blocks:
        for u_coeffs, v_coeffs, w_coeffs in terms:
            # U: A indices
            # A[i, j] -> global index i*m + j
            # Local A indices in Strassen: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                global_i = li
                global_j = a_co + lj
                U[global_i * m + global_j][current_rank] += coeff
            
            # V: B indices
            # B[i, j] -> global index i*p + j
            # Local B indices in Strassen: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                global_i = b_ro + li
                global_j = b_co + lj
                V[global_i * p + global_j][current_rank] += coeff
            
            # W: C indices
            # C[i, j] -> global index j*n + i (Trace convention W[k*n+i])
            # Local C indices in Strassen: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Note: Strassen output R is 2x2.
            # R00 corresponds to C[0, c_co]
            # R01 corresponds to C[0, c_co+1]
            # R10 corresponds to C[1, c_co]
            # R11 corresponds to C[1, c_co+1]
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                global_i = li
                global_j = c_co + lj
                W[global_j * n + global_i][current_rank] += coeff
            
            current_rank += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
