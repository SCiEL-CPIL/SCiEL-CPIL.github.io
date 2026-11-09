"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Block-level Strassen: split j into 2 groups of 2, k into 4 groups of 2
    # Each 2x2 * 2x2 sub-block uses Strassen rank 7
    # Total: 2 j-groups * 4 k-subgroups * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (2*4, 56) = (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # (4*8, 56) = (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (8*2, 56) = (16, 56)
    
    # Strassen 2x2 * 2x2:
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
    #
    # P[i,j] maps to A[i, j_global], Q[j,k] maps to B[j_global, k_global]
    # R[i,k] maps to C[i, k_global]
    
    # For each (i_local, j_local) in P: coefficient in each M
    # P00=(0,0), P01=(0,1), P10=(1,0), P11=(1,1)
    # M1: P00+P11 -> (0,0):+1, (1,1):+1
    # M2: P10+P11 -> (1,0):+1, (1,1):+1
    # M3: P00 -> (0,0):+1
    # M4: P11 -> (1,1):+1
    # M5: P00+P01 -> (0,0):+1, (0,1):+1
    # M6: P10-P00 -> (1,0):+1, (0,0):-1
    # M7: P01-P11 -> (0,1):+1, (1,1):-1
    
    # For each (j_local, k_local) in Q: coefficient in each M
    # Q00=(0,0), Q01=(0,1), Q10=(1,0), Q11=(1,1)
    # M1: Q00+Q11 -> (0,0):+1, (1,1):+1
    # M2: Q00 -> (0,0):+1
    # M3: Q01-Q11 -> (0,1):+1, (1,1):-1
    # M4: Q10-Q00 -> (1,0):+1, (0,0):-1
    # M5: Q11 -> (1,1):+1
    # M6: Q00+Q01 -> (0,0):+1, (0,1):+1
    # M7: Q10+Q11 -> (1,0):+1, (1,1):+1
    
    # For each (i_local, k_local) in R: coefficient in each M
    # R00=(0,0), R01=(0,1), R10=(1,0), R11=(1,1)
    # R00 = M1+M4-M5+M7 -> M1:+1, M4:+1, M5:-1, M7:+1
    # R01 = M3+M5 -> M3:+1, M5:+1
    # R10 = M2+M4 -> M2:+1, M4:+1
    # R11 = M1-M2+M3+M6 -> M1:+1, M2:-1, M3:+1, M6:+1
    
    # P coefficients: M_idx -> {(i_loc, j_loc): coeff}
    P_coeff = {
        0: {(0,0): 1, (1,1): 1},       # M1
        1: {(1,0): 1, (1,1): 1},       # M2
        2: {(0,0): 1},                  # M3
        3: {(1,1): 1},                  # M4
        4: {(0,0): 1, (0,1): 1},       # M5
        5: {(1,0): 1, (0,0): -1},      # M6
        6: {(0,1): 1, (1,1): -1},      # M7
    }
    
    # Q coefficients: M_idx -> {(j_loc, k_loc): coeff}
    Q_coeff = {
        0: {(0,0): 1, (1,1): 1},       # M1
        1: {(0,0): 1},                  # M2
        2: {(0,1): 1, (1,1): -1},      # M3
        3: {(1,0): 1, (0,0): -1},      # M4
        4: {(1,1): 1},                  # M5
        5: {(0,0): 1, (0,1): 1},       # M6
        6: {(1,0): 1, (1,1): 1},       # M7
    }
    
    # R coefficients: M_idx -> {(i_loc, k_loc): coeff}
    R_coeff = {
        0: {(0,0): 1, (1,1): 1},       # M1
        1: {(1,0): 1, (1,1): -1},      # M2
        2: {(0,1): 1, (1,1): 1},       # M3
        3: {(0,0): 1, (1,0): 1},       # M4
        4: {(0,1): -1, (0,1): -1},     # M5 - wait, let me redo
        5: {(1,1): 1},                  # M6
        6: {(0,0): 1, (0,1): 1},       # M7
    }
    # Let me redo R_coeff properly:
    # R00 = M1+M4-M5+M7: M1:+1, M4:+1, M5:-1, M7:+1
    # R01 = M3+M5: M3:+1, M5:+1
    # R10 = M2+M4: M2:+1, M4:+1
    # R11 = M1-M2+M3+M6: M1:+1, M2:-1, M3:+1, M6:+1
    #
    # So for each M, which R entries does it contribute to?
    # M1: R00:+1, R11:+1
    # M2: R10:+1, R11:-1
    # M3: R01:+1, R11:+1
    # M4: R00:+1, R10:+1
    # M5: R00:-1, R01:+1
    # M6: R11:+1
    # M7: R00:+1
    
    R_coeff = {
        0: {(0,0): 1, (1,1): 1},       # M1
        1: {(1,0): 1, (1,1): -1},      # M2
        2: {(0,1): 1, (1,1): 1},       # M3
        3: {(0,0): 1, (1,0): 1},       # M4
        4: {(0,0): -1, (0,1): 1},      # M5
        5: {(1,1): 1},                  # M6
        6: {(0,0): 1},                  # M7
    }
    
    # j_groups: 0 -> j in {0,1}, 1 -> j in {2,3}
    # k_subgroups: 0 -> k in {0,1}, 1 -> k in {2,3}, 2 -> k in {4,5}, 3 -> k in {6,7}
    
    col = 0
    for jg in range(2):  # j group
        j_offset = jg * 2  # 0 or 2
        for kg in range(4):  # k subgroup
            k_offset = kg * 2  # 0, 2, 4, 6
            for m_idx in range(7):  # Strassen term index
                # Fill U: for each (i_loc, j_loc) in P_coeff[m_idx]
                for (i_loc, j_loc), c_p in P_coeff[m_idx].items():
                    i_global = i_loc  # i is 0 or 1
                    j_global = j_offset + j_loc
                    U[i_global * m + j_global][col] += c_p
                
                # Fill V: for each (j_loc, k_loc) in Q_coeff[m_idx]
                for (j_loc, k_loc), c_q in Q_coeff[m_idx].items():
                    j_global = j_offset + j_loc
                    k_global = k_offset + k_loc
                    V[j_global * p + k_global][col] += c_q
                
                # Fill W: for each (i_loc, k_loc) in R_coeff[m_idx]
                for (i_loc, k_loc), c_r in R_coeff[m_idx].items():
                    i_global = i_loc  # i is 0 or 1
                    k_global = k_offset + k_loc
                    W[k_global * n + i_global][col] += c_r
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
