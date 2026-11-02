DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 (block indices)
    # C indices: 00, 01, 10, 11 (block indices)
    
    # Each Strassen term s has:
    # A_coeffs: list of (A_idx, coeff)
    # B_coeffs: list of (B_block_idx, coeff) where B_block_idx 0->cols 0-4, 1->cols 5-9
    # C_coeffs: list of (C_block_idx, coeff)
    
    strassen_terms = [
        {
            'A': [(0, 1), (3, 1)],       # A00 + A11
            'B': [(0, 1), (1, 1)],       # B00 + B11 (block 0 and block 1)
            'C': [(0, 1), (1, -1), (2, -1), (3, 1)] # C00 += M1, C11 += M1? 
            # Wait, standard Strassen:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # Let's map this to the C_coeffs list properly.
        }
    ]
    
    # Let's define the contributions explicitly for each of the 7 terms
    # Term 0: M1 = (A00+A11)(B00+B11)
    # Term 1: M2 = (A10+A11)B00
    # Term 2: M3 = A00(B01-B11)
    # Term 3: M4 = A11(B10-B00)
    # Term 4: M5 = (A00+A01)B11
    # Term 5: M6 = (A10-A00)(B00+B01)
    # Term 6: M7 = (A01-A11)(B10+B11)
    
    # C contributions:
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6
    
    # We will iterate through the 7 terms and 5 columns.
    
    col = 0
    for s in range(7):
        if s == 0:
            a_comb = [(0, 1), (3, 1)]
            b_comb = [(0, 1), (1, 1)] # B block 0 (cols 0-4) and B block 1 (cols 5-9)
            c_comb = {0: 1, 3: 1, 1: -1, 2: -1} # C00, C11, -C01? No.
            # Let's stick to the previous working logic which was verified.
            pass
            
    # Re-using the logic from the current program which was working.
    
    A_coeffs = [
        [(0, 1), (3, 1)],    # M1
        [(2, 1), (3, 1)],    # M2
        [(0, 1)],             # M3
        [(3, 1)],             # M4
        [(0, 1), (1, 1)],    # M5
        [(2, 1), (0, -1)],   # M6
        [(1, 1), (3, -1)],   # M7
    ]
    
    # B_coeffs: (block_row, block_col, coeff)
    # block_col 0 is cols 0-4, block_col 1 is cols 5-9
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1: B[0,0:5] + B[1,5:10]
        [(0, 0, 1)],               # M2: B[0,0:5]
        [(0, 1, 1), (1, 1, -1)],  # M3: B[0,5:10] - B[1,5:10]
        [(1, 0, 1), (0, 0, -1)],  # M4: B[1,0:5] - B[0,0:5]
        [(1, 1, 1)],               # M5: B[1,5:10]
        [(0, 0, 1), (0, 1, 1)],   # M6: B[0,0:5] + B[0,5:10]
        [(1, 0, 1), (1, 1, 1)],   # M7: B[1,0:5] + B[1,5:10]
    ]
    
    # C contributions: (C_block_row, C_block_col, coeff)
    # C_block 0 is cols 0-4, C_block 1 is cols 5-9
    C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1 -> C[0,0:5] += M1, C[1,5:10] += M1
        [(1, 0, 1), (1, 1, -1)],  # M2 -> C[1,0:5] += M2, C[1,5:10] -= M2
        [(0, 1, 1), (1, 1, 1)],   # M3 -> C[0,5:10] += M3, C[1,5:10] += M3
        [(0, 0, 1), (1, 0, 1)],   # M4 -> C[0,0:5] += M4, C[1,0:5] += M4
        [(0, 0, -1), (0, 1, 1)],  # M5 -> C[0,0:5] -= M5, C[0,5:10] += M5
        [(1, 1, 1)],              # M6 -> C[1,5:10] += M6
        [(0, 0, 1)],              # M7 -> C[0,0:5] += M7
    ]
    
    # Verify Strassen formulas:
    # C00 = M1 + M4 - M5 + M7. 
    # My C_coeffs:
    # M1: C00 += 1. M4: C00 += 1. M5: C00 += -1. M7: C00 += 1. Sum: 1+1-1+1 = 2? 
    # Wait. C00 should be 1.
    # Let's re-read standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check my C_coeffs against this.
    # C00 (block 0, row 0):
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # Sum: 2. This is wrong. It should be 1.
    # Ah, the standard Strassen formulas are for C = AB.
    # Let's re-verify the Strassen formulas.
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
    
    # Let's check C00 expansion:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00
    # -M5 = -A00B11 - A01B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # Sum:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # Wait, C00 should be A00B00 + A01B10.
    # My sum gives A00B00 + A01B10. Correct.
    
    # So the coefficients for C00 are:
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # Sum of coeffs for C00 is not relevant, the linear combination is what matters.
    # The W matrix accumulates these.
    # So for a specific term s, W gets the coefficient for that term.
    
    # So for M1, C00 coeff is 1.
    # For M4, C00 coeff is 1.
    # For M5, C00 coeff is -1.
    # For M7, C00 coeff is 1.
    
    # My C_coeffs list above:
    # Term 0 (M1): C00: 1, C11: 1. (Correct: M1 is in C00 and C11)
    # Term 1 (M2): C10: 1, C11: -1. (Correct: M2 is in C10 and C11)
    # Term 2 (M3): C01: 1, C11: 1. (Correct: M3 is in C01 and C11)
    # Term 3 (M4): C00: 1, C10: 1. (Correct: M4 is in C00 and C10)
    # Term 4 (M5): C00: -1, C01: 1. (Correct: M5 is in C00 and C01)
    # Term 5 (M6): C11: 1. (Correct: M6 is in C11)
    # Term 6 (M7): C00: 1. (Correct: M7 is in C00)
    
    # This looks correct.
    
    for s in range(7):
        for t in range(5):
            # U
            for a_idx, a_c in A_coeffs[s]:
                U[a_idx][col] = a_c
            
            # V
            for b_row, b_block, b_c in B_coeffs[s]:
                b_col = b_block * 5 + t
                V[b_row * p + b_col][col] = b_c
            
            # W
            for c_row, c_block, c_c in C_coeffs[s]:
                c_col = c_block * 5 + t
                w_idx = c_col * n + c_row
                W[w_idx][col] = c_c
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
