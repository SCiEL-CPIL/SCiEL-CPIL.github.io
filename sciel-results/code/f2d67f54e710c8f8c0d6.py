"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # We decompose into 8 independent 2x2x2 Strassen blocks.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 rows
    V = [[0] * rank for _ in range(m * p)] # 32 rows
    W = [[0] * rank for _ in range(p * n)] # 16 rows (k*n+i)
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # A indices: (i, j) in 0,1
    # B indices: (j, k) in 0,1
    # C indices: (i, k) in 0,1
    
    # For each M_m, we store:
    # U_coeffs: dict {(i,j): coeff}
    # V_coeffs: dict {(j,k): coeff}
    # W_coeffs: dict {(i,k): coeff}
    
    strassen_blocks = [
        # M1
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1} # Wait, W is for reconstruction.
        }
    ]
    # Actually, W coefficients depend on how M_m is used in C reconstruction.
    # So we should iterate over C entries and see which M_m contribute.
    # Or iterate over M_m and add to W.
    
    # Let's define the Strassen steps explicitly.
    # Step m:
    # U_part: linear combo of A
    # V_part: linear combo of B
    # W_part: linear combo of C (reconstruction)
    
    # It is easier to define for each m:
    # A_terms: list of (i, j, coeff)
    # B_terms: list of (j, k, coeff)
    # C_terms: list of (i, k, coeff) -- these are the contributions to C.
    
    # M1
    # A: (0,0,1), (1,1,1)
    # B: (0,0,1), (1,1,1)
    # C: (0,0,1), (1,1,1)
    
    # M2
    # A: (1,0,1), (1,1,1)
    # B: (0,0,1)
    # C: (1,0,1), (1,1,-1)
    
    # M3
    # A: (0,0,1)
    # B: (1,0,1), (1,1,-1)
    # C: (0,1,1), (1,1,1)
    
    # M4
    # A: (1,1,1)
    # B: (0,1,1), (1,0,-1)
    # C: (0,0,1), (1,0,1)
    
    # M5
    # A: (0,0,1), (0,1,1)
    # B: (1,1,1)
    # C: (0,0,-1), (0,1,1)
    
    # M6
    # A: (1,0,1), (0,0,-1)
    # B: (0,0,1), (1,0,1)
    # C: (1,1,1)
    
    # M7
    # A: (0,1,1), (1,1,-1)
    # B: (0,1,1), (1,1,1)
    # C: (0,0,1)
    
    # Let's verify C reconstruction:
    # C00 = M1 + M4 - M5 + M7
    #   M1: 1
    #   M4: 1
    #   M5: -1
    #   M7: 1
    #   Sum: 1+1-1+1 = 2? No.
    #   Strassen C00 = M1 + M4 - M5 + M7.
    #   If M1=1, M4=1, M5=1, M7=1, C00=2.
    #   Correct formula: C00 = M1 + M4 - M5 + M7.
    #   My W coeffs above:
    #   M1: C00=1
    #   M4: C00=1
    #   M5: C00=-1
    #   M7: C00=1
    #   Sum: 1+1-1+1 = 2.
    #   Wait, is C00 = M1 + M4 - M5 + M7?
    #   Yes.
    
    # C01 = M3 + M5
    #   M3: C01=1
    #   M5: C01=1
    #   Sum: 2.
    
    # C10 = M2 + M4
    #   M2: C10=1
    #   M4: C10=1
    #   Sum: 2.
    
    # C11 = M1 - M2 + M3 + M6
    #   M1: C11=1
    #   M2: C11=-1
    #   M3: C11=1
    #   M6: C11=1
    #   Sum: 1-1+1+1 = 2.
    
    # So the coefficients in W should be as defined.
    
    strassen_defs = [
        # m, A_terms, B_terms, C_terms
        (0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        (1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),           # M2
        (2, [(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),           # M3
        (3, [(1,1,1)], [(0,1,1), (1,0,-1)], [(0,0,1), (1,0,1)]),           # M4
        (4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),           # M5
        (5, [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),           # M6
        (6, [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(0,0,1)]),           # M7
    ]
    
    # Iterate over the 8 blocks
    for s in range(8):
        w = s // 2 # 0..3
        t = s % 2  # 0,1
        
        # Global index mappings
        # A: (i_loc, j_loc) -> (i, j)
        # i = i_loc
        # j = j_loc + 2*t
        
        # B: (j_loc, k_loc) -> (j, k)
        # j = j_loc + 2*t
        # k = k_loc + 2*w
        
        # C: (i_loc, k_loc) -> (i, k)
        # i = i_loc
        # k = k_loc + 2*w
        
        for m_idx, (m, A_terms, B_terms, C_terms) in enumerate(strassen_defs):
            c = s * 7 + m_idx
            
            # Fill U
            for (i_loc, j_loc, coeff) in A_terms:
                i = i_loc
                j = j_loc + 2 * t
                U[i * m + j][c] += coeff
            
            # Fill V
            for (j_loc, k_loc, coeff) in B_terms:
                j = j_loc + 2 * t
                k = k_loc + 2 * w
                V[j * p + k][c] += coeff
            
            # Fill W
            # W index is k*n + i
            for (i_loc, k_loc, coeff) in C_terms:
                i = i_loc
                k = k_loc + 2 * w
                W[k * n + i][c] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
