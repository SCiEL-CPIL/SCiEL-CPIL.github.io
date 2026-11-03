"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # Inputs A (2x2), B (2x2)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Output P (2x2)
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M5
    # P21 = M2 + M4
    # P22 = M1 - M2 + M3 + M6
    
    # We define the 7 terms as lists of (coeff_A, coeff_B, coeff_P)
    # For each term, we specify which local A indices, B indices, and P indices are involved.
    # Local indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # P: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_terms = [
        # (A_indices, B_indices, P_indices)
        # M1: (A00+A11) * (B00+B11) -> P00, P11
        ([0, 3], [0, 3], [0, 3]), # Coeffs: A: 1,1; B: 1,1; P: 1,1
        # M2: (A10+A11) * B00 -> P10
        ([2, 3], [0], [2]), # Coeffs: A: 1,1; B: 1; P: 1
        # M3: A00 * (B01-B11) -> P01
        ([0], [1, 3], [1]), # Coeffs: A: 1; B: 1,-1; P: 1
        # M4: A11 * (B10-B00) -> P00, P10
        ([3], [2, 0], [0, 2]), # Coeffs: A: 1; B: 1,-1; P: 1,1
        # M5: (A00+A01) * B11 -> P00, P01
        ([0, 1], [3], [0, 1]), # Coeffs: A: 1,1; B: 1; P: -1, 1
        # M6: (A10-A00) * (B00+B01) -> P11
        ([2, 0], [0, 1], [3]), # Coeffs: A: 1,-1; B: 1,1; P: 1
        # M7: (A01-A11) * (B10+B11) -> P00, P11
        ([1, 3], [2, 3], [0, 3]), # Coeffs: A: 1,-1; B: 1,1; P: 1,1
    ]
    
    # Define the 4 blocks
    # Block 1: A1 (cols 0-1), B11 (rows 0-1, cols 0-1) -> C cols 0-1
    # Block 2: A2 (cols 2-3), B21 (rows 2-3, cols 0-1) -> C cols 0-1
    # Block 3: A1 (cols 0-1), B12 (rows 0-1, cols 2-3) -> C cols 2-3
    # Block 4: A2 (cols 2-3), B22 (rows 2-3, cols 2-3) -> C cols 2-3
    
    # Mapping local indices to global indices
    # U index: i*m + j. n=2, m=4.
    # A1: i=0,1; j=0,1. U indices: 0,1, 4,5? No.
    # U[i*m + j].
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # A2: i=0,1; j=2,3.
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # V index: j*p + k. m=4, p=4.
    # B11: j=0,1; k=0,1.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    # B12: j=0,1; k=2,3.
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    # B21: j=2,3; k=0,1.
    # j=2, k=0 -> 8
    # j=2, k=1 -> 9
    # j=3, k=0 -> 12
    # j=3, k=1 -> 13
    # B22: j=2,3; k=2,3.
    # j=2, k=2 -> 10
    # j=2, k=3 -> 11
    # j=3, k=2 -> 14
    # j=3, k=3 -> 15
    
    # W index: k*n + i. p=4, n=2.
    # C cols 0-1 (from B11, B21): k=0,1.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # C cols 2-3 (from B12, B22): k=2,3.
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    
    blocks = [
        # (U_offset, V_offset, W_offset)
        # Block 1: A1, B11, C01
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3]),
        # Block 2: A2, B21, C01
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3]),
        # Block 3: A1, B12, C23
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]),
        # Block 4: A2, B22, C23
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7]),
    ]
    
    col = 0
    for u_off, v_off, w_off in blocks:
        for a_idx, b_idx, p_idx in strassen_terms:
            # a_idx is list of local indices
            # We need to handle coefficients.
            # The strassen_terms defined above are just indices.
            # We need to hardcode the coefficients for each term.
            
            # Let's redefine terms with coefficients explicitly.
            # Term 0: M1
            if a_idx == [0, 3] and b_idx == [0, 3] and p_idx == [0, 3]:
                a_coeffs = [1, 1]
                b_coeffs = [1, 1]
                p_coeffs = [1, 1]
            elif a_idx == [2, 3] and b_idx == [0] and p_idx == [2]:
                a_coeffs = [1, 1]
                b_coeffs = [1]
                p_coeffs = [1]
            elif a_idx == [0] and b_idx == [1, 3] and p_idx == [1]:
                a_coeffs = [1]
                b_coeffs = [1, -1]
                p_coeffs = [1]
            elif a_idx == [3] and b_idx == [2, 0] and p_idx == [0, 2]:
                a_coeffs = [1]
                b_coeffs = [1, -1]
                p_coeffs = [1, 1]
            elif a_idx == [0, 1] and b_idx == [3] and p_idx == [0, 1]:
                a_coeffs = [1, 1]
                b_coeffs = [1]
                p_coeffs = [-1, 1]
            elif a_idx == [2, 0] and b_idx == [0, 1] and p_idx == [3]:
                a_coeffs = [1, -1]
                b_coeffs = [1, 1]
                p_coeffs = [1]
            elif a_idx == [1, 3] and b_idx == [2, 3] and p_idx == [0, 3]:
                a_coeffs = [1, -1]
                b_coeffs = [1, 1]
                p_coeffs = [1, 1]
            else:
                raise ValueError("Unknown term")
            
            for i, idx in enumerate(a_idx):
                U[u_off[idx]][col] = a_coeffs[i]
            for i, idx in enumerate(b_idx):
                V[v_off[idx]][col] = b_coeffs[i]
            for i, idx in enumerate(p_idx):
                W[w_off[idx]][col] = p_coeffs[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
