"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 4 independent 2x2x2 Strassen decompositions for the 4 blocks of B
    # B is 2x8, split into four 2x2 blocks.
    # Rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Contributions to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear forms for A (u), B (v), C (w) for each of the 7 terms
    # A is flattened 00, 01, 10, 11 -> indices 0, 1, 2, 3 (i*m+j, m=2)
    # B is flattened 00, 01, 10, 11 -> indices 0, 1, 2, 3 (j*p+k, p=8, but relative to block)
    # C is flattened 00, 01, 10, 11 -> indices 0, 1, 2, 3 (k*n+i, n=2, but relative to block)
    
    # We need to map relative indices to global indices.
    # Block b corresponds to B columns 2b, 2b+1 and C columns 2b, 2b+1.
    
    def get_A_idx(i, j):
        return i * m + j
        
    def get_B_idx(j, k_rel, b):
        k_global = 2 * b + k_rel
        return j * p + k_global
        
    def get_C_idx(k_rel, i, b):
        k_global = 2 * b + k_rel
        return k_global * n + i
        
    # Term definitions: (u_coeffs, v_coeffs, w_coeffs)
    # u: dict {A_idx: coeff}
    # v: dict {B_rel_idx: coeff}
    # w: dict {C_rel_idx: coeff}
    
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 2: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # Wait, C10=M2+M4, C11=M1-M2...
        # Let's re-verify Strassen C formulas
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 contributes to C00 (+1) and C11 (+1)
        # M2 contributes to C10 (+1) and C11 (-1)
        # M3 contributes to C01 (+1) and C11 (+1)
        # M4 contributes to C00 (+1) and C10 (+1)
        # M5 contributes to C00 (-1) and C01 (+1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        # Corrected W coefficients:
        # M1: C00(+1), C11(+1)
        # M2: C10(+1), C11(-1)
        # M3: C01(+1), C11(+1)
        # M4: C00(+1), C10(+1)
        # M5: C00(-1), C01(+1)
        # M6: C11(+1)
        # M7: C00(+1)
    ]
    
    # Re-defining terms with correct W
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 2: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: A00(B10-B11)
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A11(B01-B00)
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {2: 1}, {0: -1, 1: 1}),
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for b in range(4):
        for u_dict, v_dict, w_dict in strassen_terms:
            # Fill U
            for a_idx, coeff in u_dict.items():
                U[a_idx][col] = coeff
                
            # Fill V
            for b_rel_idx, coeff in v_dict.items():
                # b_rel_idx is 0..3 corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
                # Map to global B index
                j_rel = b_rel_idx // 2
                k_rel = b_rel_idx % 2
                v_idx = get_B_idx(j_rel, k_rel, b)
                V[v_idx][col] = coeff
                
            # Fill W
            for c_rel_idx, coeff in w_dict.items():
                # c_rel_idx is 0..3 corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
                # Map to global C index
                i_rel = c_rel_idx // 2
                k_rel = c_rel_idx % 2
                w_idx = get_C_idx(k_rel, i_rel, b)
                W[w_idx][col] = coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
