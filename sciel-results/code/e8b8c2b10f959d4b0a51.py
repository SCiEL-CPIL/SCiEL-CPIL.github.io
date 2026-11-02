"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using block Strassen: split B into 7 blocks of 2x2
    # Total rank = 7 blocks * 7 Strassen terms = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A indices: 00, 01, 10, 11 -> U rows: 0, 1, 2, 3
    # Strassen terms (X, Y, Z)
    # X is linear combo of A, Y is linear combo of B, Z is linear combo of C
    # We define the 7 terms for a generic 2x2 * 2x2
    # Local B indices: 00, 01, 10, 11
    # Local C indices: 00, 01, 10, 11
    
    # Term definitions:
    # 1: (A00+A11) * (B00+B11) -> C00 += 1, C11 += 1
    # 2: (A10+A11) * B00 -> C10 += 1, C11 -= 1
    # 3: A00 * (B01-B11) -> C01 += 1, C11 += 1
    # 4: A11 * (B10-B00) -> C00 += 1, C10 += 1
    # 5: (A00+A01) * B11 -> C00 -= 1, C01 += 1
    # 6: (A10-A00) * (B00+B01) -> C11 += 1
    # 7: (A01-A11) * (B10+B11) -> C00 += 1, C10 -= 1
    
    # Let's verify Strassen formulas again to be sure.
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
    
    # Let's map these to (A_coeffs, B_coeffs, C_coeffs)
    # A coeffs: [A00, A01, A10, A11]
    # B coeffs: [B00, B01, B10, B11]
    # C coeffs: [C00, C01, C10, C11]
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, -1, 0]), # M7
    ]
    
    # Map local B index (00, 01, 10, 11) to global B row index (j*14 + k)
    # Local 00 -> j=0, k=2b
    # Local 01 -> j=0, k=2b+1
    # Local 10 -> j=1, k=2b
    # Local 11 -> j=1, k=2b+1
    
    # Map local C index (00, 01, 10, 11) to global C row index (k*n + i)
    # Local 00 -> i=0, k=2b -> row 2b*2 + 0
    # Local 01 -> i=0, k=2b+1 -> row (2b+1)*2 + 0
    # Local 10 -> i=1, k=2b -> row 2b*2 + 1
    # Local 11 -> i=1, k=2b+1 -> row (2b+1)*2 + 1
    
    # Map local A index (00, 01, 10, 11) to global A row index (i*m + j)
    # Local 00 -> i=0, j=0 -> row 0
    # Local 01 -> i=0, j=1 -> row 1
    # Local 10 -> i=1, j=0 -> row 2
    # Local 11 -> i=1, j=1 -> row 3
    
    for b in range(7):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        for t_idx in range(7):
            r = b * 7 + t_idx
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[t_idx]
            
            # Fill U
            # a_coeffs are for A00, A01, A10, A11
            # U rows: 0, 1, 2, 3
            for a_idx in range(4):
                if a_coeffs[a_idx] != 0:
                    U[a_idx][r] = a_coeffs[a_idx]
                    
            # Fill V
            # b_coeffs are for B00, B01, B10, B11
            # B00 -> row 0*14 + k1
            # B01 -> row 0*14 + k2
            # B10 -> row 1*14 + k1
            # B11 -> row 1*14 + k2
            b_rows = [0 * 14 + k1, 0 * 14 + k2, 1 * 14 + k1, 1 * 14 + k2]
            for b_idx in range(4):
                if b_coeffs[b_idx] != 0:
                    V[b_rows[b_idx]][r] = b_coeffs[b_idx]
                    
            # Fill W
            # c_coeffs are for C00, C01, C10, C11
            # C00 -> row k1*2 + 0
            # C01 -> row k2*2 + 0
            # C10 -> row k1*2 + 1
            # C11 -> row k2*2 + 1
            c_rows = [k1 * 2 + 0, k2 * 2 + 0, k1 * 2 + 1, k2 * 2 + 1]
            for c_idx in range(4):
                if c_coeffs[c_idx] != 0:
                    W[c_rows[c_idx]][r] = c_coeffs[c_idx]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
