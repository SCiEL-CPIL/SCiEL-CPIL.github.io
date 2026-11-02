def construct():
    n, m, p = 4, 4, 7
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # Each term is (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: [a00, a01, a10, a11]
    # b_coeffs: [b00, b01, b10, b11]
    # c_coeffs: [c00, c01, c10, c11]
    
    # We define the 7 Strassen terms for 2x2.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to express each Mk as a sum of products of linear combinations.
    # Actually, each Mk IS a product of two linear combinations.
    # So we just need the coefficients for A and B for each Mk.
    
    # M1: A: [1,0,0,1], B: [1,0,0,1]
    # M2: A: [0,0,1,1], B: [1,0,0,0]
    # M3: A: [1,0,0,0], B: [0,1,0,-1]
    # M4: A: [0,0,0,1], B: [0,0,1,-1]
    # M5: A: [1,1,0,0], B: [0,0,0,1]
    # M6: A: [0,0,1,-1], B: [1,1,0,0]
    # M7: A: [0,1,0,-1], B: [0,0,1,1]
    
    # And the contribution to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the 4x4 case, we apply this to the 2x2 blocks.
    # We iterate over the 7 outer terms.
    # For each outer term, we have a 2x2 matrix X and Y.
    # We apply the 7 inner Strassen terms to X and Y.
    
    # This is a 7x7 = 49 term decomposition.
    
    # I will implement this logic.
    
    return {"U": U, "V": V, "W": W}
