"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Block-level Strassen:
    # Split m (4) into 2 groups of 2 (J0, J1)
    # Split p (8) into 4 groups of 2 (K0, K1, K2, K3)
    # n (2) is not split.
    # We have 2 * 4 = 8 blocks of 2x2x2 multiplication.
    # Each block uses Strassen rank 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (16, 56)
    
    # Strassen coefficients for 2x2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define coefficients for A, B, C for each Strassen term
    # Indexing: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A_coeff[m_idx][local_idx]
    # B_coeff[m_idx][local_idx]
    # C_coeff[m_idx][local_idx]
    
    A_coeff = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [1, 0, 1, 0],  # M6: A10 - A00 (wait, A10 is (1,0), A00 is (0,0))
                       # My local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                       # So A10 is idx 2, A00 is idx 0.
                       # M6 = (A10 - A00). Coeffs: 0:-1, 2:1.
                       # Let's re-verify.
                       # M6 = (A10 - A00)(B00 + B01)
                       # A part: A10 (1) - A00 (1).
                       # So coeff for 0 is -1, coeff for 2 is 1.
        [0, 1, 0, -1], # M7: A01 - A11. A01 is idx 1, A11 is idx 3.
    ]
    
    # Correction for A_coeff M6:
    A_coeff[5] = [-1, 0, 1, 0]
    
    B_coeff = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    
    C_coeff = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11
        [0, 1, 0, 1],  # M3: C01 + C11
        [1, 0, 1, 0],  # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0],  # M7: C00
    ]
    
    col = 0
    # j_groups: 0 -> j in {0,1}, 1 -> j in {2,3}
    # k_groups: 0 -> k in {0,1}, 1 -> k in {2,3}, 2 -> k in {4,5}, 3 -> k in {6,7}
    
    for jg in range(2):
        j_offset = jg * 2
        for kg in range(4):
            k_offset = kg * 2
            for m_idx in range(7):
                # Fill U (A part)
                for i_loc in range(2):
                    for j_loc in range(2):
                        c = A_coeff[m_idx][i_loc * 2 + j_loc]
                        if c != 0:
                            i_global = i_loc
                            j_global = j_offset + j_loc
                            U[i_global * m + j_global][col] += c
                
                # Fill V (B part)
                for j_loc in range(2):
                    for k_loc in range(2):
                        c = B_coeff[m_idx][j_loc * 2 + k_loc]
                        if c != 0:
                            j_global = j_offset + j_loc
                            k_global = k_offset + k_loc
                            V[j_global * p + k_global][col] += c
                
                # Fill W (C part)
                for i_loc in range(2):
                    for k_loc in range(2):
                        c = C_coeff[m_idx][i_loc * 2 + k_loc]
                        if c != 0:
                            i_global = i_loc
                            k_global = k_offset + k_loc
                            W[k_global * n + i_global][col] += c
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
