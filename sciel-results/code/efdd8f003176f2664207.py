DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the 2x4 x 4x4 case.
    # A1 is 2x4. A1 = [A11, A12] where A11, A12 are 2x2.
    # B is 4x4. B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C1 is 2x4. C1 = [C11, C12] where C11, C12 are 2x2.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    
    # We compute C11 and C12 separately using Strassen?
    # No, we can compute them using the 7 Strassen products for 2x2.
    # But C11 and C12 share the A blocks.
    # Actually, we can just use 2 independent Strassen decompositions?
    # One for C11 = A1 B_L (where B_L is 4x2).
    # One for C12 = A1 B_R (where B_R is 4x2).
    
    # Let's decompose 2x4 x 4x2 multiplication.
    # A is 2x4. B is 4x2. C is 2x2.
    # This is equivalent to 2x2 x 2x2 if we view A as two 2x2 blocks?
    # A = [A_L, A_R]. B = [B_T; B_B].
    # C = A_L B_T + A_R B_B.
    # This is a sum of two 2x2 products.
    # Rank 7 + 7 = 14.
    
    # So for C11 (2x2), we use 14 terms.
    # For C12 (2x2), we use 14 terms.
    # Total 28 terms for C1.
    
    # For C2 (1x4), we use 4 terms.
    
    # Total 32 terms.
    
    return {"U": U, "V": V, "W": W}

