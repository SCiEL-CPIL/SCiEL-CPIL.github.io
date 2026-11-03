import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 5
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a term
    # u_indices: list of (idx, coeff)
    # v_indices: list of (idx, coeff)
    # w_indices: list of (idx, coeff)
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, coeff in u_indices:
            U[idx][col] = coeff
        for idx, coeff in v_indices:
            V[idx][col] = coeff
        for idx, coeff in w_indices:
            W[idx][col] = coeff
        col += 1

    # Part 1: Strassen 4x4x4
    # Blocks
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B00: 0,1,5,6
    # B01: 2,3,7,8
    # B10: 10,11,15,16
    # B11: 12,13,17,18
    
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15

    # Strassen products
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    # M2 = (A10 + A11)B00 -> C10
    # M3 = A00(B10 - B11) -> C01, C11
    # M4 = A11(B01 - B00) -> C00, C10
    # M5 = (A00 + A01)B11 -> C00, C01
    # M6 = (A10 - A00)(B00 + B01) -> C11
    # M7 = (A01 - A11)(B10 + B11) -> C00

    # We need to define the 4 schoolbook terms for each M_k.
    # For a product X*Y, the terms are:
    # 1. X00*Y00 -> Z00
    # 2. X01*Y10 -> Z00
    # 3. X00*Y01 -> Z01
    # 4. X01*Y11 -> Z01
    # 5. X10*Y00 -> Z10
    # 6. X11*Y10 -> Z10
    # 7. X10*Y01 -> Z11
    # 8. X11*Y11 -> Z11
    # Wait, schoolbook 2x2 has 4 multiplications?
    # No, schoolbook 2x2 is 4 multiplications if we are lucky?
    # No, standard schoolbook 2x2 is 4 multiplications?
    # C00 = A00B00 + A01B10 (2 mults)
    # C01 = A00B01 + A01B11 (2 mults)
    # C10 = A10B00 + A11B10 (2 mults)
    # C11 = A10B01 + A11B11 (2 mults)
    # Total 8 multiplications.
    # But Strassen 4x4x4 rank is 28.
    # 7 products. 28 / 7 = 4.
    # So each 2x2 multiplication must be done in rank 4.
    # Is rank 4 possible for 2x2x2?
    # Yes, the border rank is 4? No, rank is 7.
    # Wait. Strassen 4x4x4 rank is 28?
    # If we use Strassen for 2x2x2 (rank 7), total is 49.
    # If we use schoolbook for 2x2x2 (rank 8? No, 4 mults is not possible for 2x2x2).
    # The rank of 2x2x2 is 7.
    # So 7 * 7 = 49.
    # But I said 28.
    # Let's check.
    # Rank of 4x4x4 is 28?
    # Yes, it is known that rank(4x4x4) <= 28.
    # This is achieved by a specific algorithm, not just Strassen with schoolbook base.
    # Actually, Strassen 4x4x4 using schoolbook 2x2 (which is rank 8? No, schoolbook 2x2 is 4 mults? No, 8 terms? No, 4 mults is wrong).
    # Schoolbook 2x2:
    # M1 = A00B00
    # M2 = A01B10
    # M3 = A00B01
    # M4 = A01B11
    # M5 = A10B00
    # M6 = A11B10
    # M7 = A10B01
    # M8 = A11B11
    # This is rank 8.
    # So 7 * 8 = 56.
    # But rank 28 is known.
    # How?
    # It uses a different decomposition.
    # I will stick to Rank 56 + 16 = 72?
    # Or Rank 49 + 16 = 65?
    # Or just Schoolbook 80?
    # 72 is better than 80.
    # Let's implement Strassen 4x4x4 with Schoolbook 2x2 (Rank 8 base).
    # Total 56.
    # Plus 16 for 4x4x1.
    # Total 72.

    # Wait, is there a rank 4 decomposition for 2x2x2?
    # No, min rank is 7.
    # So Strassen 4x4x4 is 49.
    # 49 + 16 = 65.
    
    # Let's implement Strassen 4x4x4 (Rank 49) + Schoolbook 4x4x1 (Rank 16).
    # Total 65.

    # Reset rank
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Strassen 2x2x2 Rank 7 terms
    # S1 = A00 + A11, T1 = B00 + B11 -> P1
    # S2 = A10 + A11, T2 = B00 -> P2
    # S3 = A00, T3 = B10 - B11 -> P3
    # S4 = A11, T4 = B01 - B00 -> P4
    # S5 = A00 + A01, T5 = B11 -> P5
    # S6 = A10 - A00, T6 = B00 + B01 -> P6
    # S7 = A01 - A11, T7 = B10 + B11 -> P7
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # For each Pi, we add 1 term?
    # No, Pi is a scalar product of two linear combinations.
    # So each Pi is 1 rank-1 term in the U,V,W sense?
    # Yes!
    # U has the coeffs of S_k.
    # V has the coeffs of T_k.
    # W has the coeffs for C based on the formula.
    
    # So Strassen 4x4x4 has 7 terms?
    # No, the 2x2 matrices are multiplied.
    # The result is a 2x2 matrix.
    # The elements of the result are linear combinations of the Pi.
    # So for each Pi, we add 4 terms to W?
    # No.
    # Pi is a scalar.
    # Pi contributes to C00, C01, C10, C11.
    # So for each Pi, we add 1 term to U, V.
    # And we add 1 term to W for each C element it affects.
    # But W is a matrix.
    # The term is rank-1.
    # U[:, k] is fixed for Pi.
    # V[:, k] is fixed for Pi.
    # W[:, k] must be the vector of contributions.
    # So for each Pi, we have 1 column in U, V.
    # And 1 column in W with non-zeros at the affected C elements.
    
    # So Strassen 4x4x4 uses 7 terms?
    # No, the rank of 4x4x4 is 28?
    # If we use 7 terms, rank is 7.
    # But 4x4x4 rank is 28.
    # Why?
    # Because the 2x2 matrices are not scalars.
    # The "multiplication" is of matrices.
    # So we need to decompose the matrix multiplication tensor.
    # The tensor for 4x4x4 is the sum of 7 tensors of type 2x2x2.
    # Each 2x2x2 tensor has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # So we need to expand each Pi into 7 rank-1 terms (Strassen 2x2).
    # Total 49 terms.

    # Let's implement Strassen 2x2 for each of the 7 products.
    
    def add_strassen_2x2(x_coeffs, y_coeffs, z_coeffs, col_start):
        # x_coeffs: dict {(i,j): coeff} for 2x2
        # y_coeffs: dict {(i,j): coeff} for 2x2
        # z_coeffs: dict {(i,j): coeff} for 2x2 (output)
        
        # Strassen 2x2 terms
        # S1 = x00 + x11, T1 = y00 + y11
        # S2 = x10 + x11, T2 = y00
        # S3 = x00, T3 = y10 - y11
        # S4 = x11, T4 = y01 - y00
        # S5 = x00 + x01, T5 = y11
        # S6 = x10 - x00, T6 = y00 + y01
        # S7 = x01 - x11, T7 = y10 + y11
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We add 7 terms.
        
        # Term 1
        u1 = {}
        for (i,j), c in x_coeffs.items():
            if (i,j) == (0,0) or (i,j) == (1,1):
                u1[(i,j)] = c
        v1 = {}
        for (i,j), c in y_coeffs.items():
            if (i,j) == (0,0) or (i,j) == (1,1):
                v1[(i,j)] = c
        w1 = {}
        for (i,j), c in z_coeffs.items():
            if (i,j) == (0,0): w1[(i,j)] = c
            if (i,j) == (1,1): w1[(i,j)] = c
            
        # This is getting complicated to map to global indices.
        # I will hardcode the 49 terms for Strassen 4x4x4.
        
        pass

    # Given the complexity, I will fallback to Rank 80 Schoolbook.
    # It is the only one I can guarantee to be correct quickly.
    # But 80 is score 1.0.
    # I need to beat 1.0.
    # Let's try Rank 72 (Strassen 4x4x4 with Schoolbook 2x2 + Schoolbook 4x4x1).
    
    # Rank 72 implementation.
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Strassen 4x4x4 with Schoolbook 2x2 (Rank 8 base)
    # 7 products * 8 terms = 56 terms.
    
    # Helper to add a schoolbook 2x2 product
    def add_schoolbook_2x2(x_coeffs, y_coeffs, z_coeffs, col_start):
        nonlocal col
        # x_coeffs: {(i,j): coeff}
        # y_coeffs: {(i,j): coeff}
        # z_coeffs: {(i,j): coeff}
        
        # Terms:
        # 1. x00*y00 -> z00
        # 2. x01*y10 -> z00
        # 3. x00*y01 -> z01
        # 4. x01*y11 -> z01
        # 5. x10*y00 -> z10
        # 6. x11*y10 -> z10
        # 7. x10*y01 -> z11
        # 8. x11*y11 -> z11
        
        terms = [
            ((0,0), (0,0), (0,0)),
            ((0,1), (1,0), (0,0)),
            ((0,0), (0,1), (0,1)),
            ((0,1), (1,1), (0,1)),
            ((1,0), (0,0), (1,0)),
            ((1,1), (1,0), (1,0)),
            ((1,0), (0,1), (1,1)),
            ((1,1), (1,1), (1,1))
        ]
        
        for (xi, xj), (yi, yj), (zi, zj) in terms:
            cx = x_coeffs.get((xi, xj), 0)
            cy = y_coeffs.get((yi, yj), 0)
            cz = z_coeffs.get((zi, zj), 0)
            
            if cx != 0 and cy != 0 and cz != 0:
                U_idx = xi * 4 + xj # This is local 2x2 index. Need global.
                # This function needs to know the offset.
                pass

    # This is too hard to do generically.
    # I will return Rank 80.
    
    U = [[0] * 80 for _ in range(n * m)]
    V = [[0] * 80 for _ in range(m * p)]
    W = [[0] * 80 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
