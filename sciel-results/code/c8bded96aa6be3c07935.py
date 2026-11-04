"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition using a recursive Strassen-like approach.
    # 1. Decompose 3x3x3 into a 2x2x2 Strassen part (rank 7) and lower-rank corrections.
    # 2. Decompose the remaining 3x3x1 part (rank 3).
    # Total rank <= 7 (Strassen 2x2) + 9 (corrections for 3x3) + 3 (for 4th col) ?
    # Let's refine the rank count for 3x3x3 using Strassen on the 2x2 block.
    
    # The 3x3x3 multiplication can be viewed as:
    # A = [[A22, a12], [a21, a33]] where A22 is 2x2.
    # B = [[B22, b12], [b21, b33]] where B22 is 2x2.
    # C = [[A22B22 + a12b21, A22b12 + a12b33],
    #      [a21B22 + a33b21, a21b12 + a33b33]]
    
    # Term 1: A22B22 (2x2x2). Rank 7 (Strassen).
    # Term 2: a12b21 (1x1x1). Rank 1.
    # Term 3: A22b12 (2x2x1). Rank 2.
    # Term 4: a12b33 (1x1x1). Rank 1.
    # Term 5: a21B22 (1x2x2). Rank 2.
    # Term 6: a33b21 (1x1x1). Rank 1.
    # Term 7: a21b12 (1x1x1). Rank 1.
    # Term 8: a33b33 (1x1x1). Rank 1.
    
    # Total Rank for 3x3x3 = 7 + 1 + 2 + 1 + 2 + 1 + 1 + 1 = 16.
    
    # Now for 3x3x4:
    # B is 3x4. Split into B_3x3 and B_3x1.
    # C = A * B_3x3 + A * B_3x1.
    # Rank(A * B_3x3) <= 16.
    # Rank(A * B_3x1): A is 3x3, B is 3x1. Output 3x1.
    # This is 3x3x1. Rank 3.
    # Total Rank = 16 + 3 = 19.
    
    rank = 19
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # --- Part 1: 3x3x3 Decomposition ---
    
    # Indices for A (3x3): 0..8 (row-major)
    # Indices for B (3x3 part): 0..8 (row-major, cols 0..2)
    # Indices for C (3x3 part): 0..8 (row-major, cols 0..2)
    
    # 1. Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1,3,4 (00,01,10,11)
    # B indices: 0,1,3,4
    # C indices: 0,1,3,4
    
    # Strassen Terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 1, 1) # A00, B00, C00
    add_term(4, 4, 0, 1, 1, 1) # A11, B11, C00
    add_term(0, 0, 4, 1, 1, 1) # A00, B00, C11
    add_term(4, 4, 4, 1, 1, 1) # A11, B11, C11
    
    # Wait, Strassen is 7 terms, not 4. The above is schoolbook for 2x2.
    # I need to implement the actual Strassen coefficients.
    
    # Let's restart the term addition with a helper that takes explicit U,V,W vectors.
    
    # Clear previous attempts
    col = 0
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_rank1(u_vec, v_vec, w_vec):
        """
        u_vec: dict or list of (idx, val) for U
        v_vec: dict or list of (idx, val) for V
        w_vec: dict or list of (idx, val) for W
        """
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_vec:
            U[idx][col] += val
        for idx, val in v_vec:
            V[idx][col] += val
        for idx, val in w_vec:
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 Coefficients
    # A indices: 0(00), 1(01), 2(10), 3(11) relative to 2x2 block.
    # Global A indices for top-left 2x2: 0,1,3,4.
    # Global B indices for top-left 2x2: 0,1,3,4.
    # Global C indices for top-left 2x2: 0,1,3,4.
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    u1 = [(0, 1), (4, 1)]
    v1 = [(0, 1), (4, 1)]
    w1 = [(0, 1), (4, 1)]
    add_rank1(u1, v1, w1)
    
    # M2 = (A10+A11)B00
    # C10 += M2
    u2 = [(3, 1), (4, 1)]
    v2 = [(0, 1)]
    w2 = [(3, 1)]
    add_rank1(u2, v2, w2)
    
    # M3 = A00(B01-B11)
    # C01 += M3
    u3 = [(0, 1)]
    v3 = [(1, 1), (4, -1)]
    w3 = [(1, 1)]
    add_rank1(u3, v3, w3)
    
    # M4 = A11(B10-B00)
    # C10 += M4
    u4 = [(4, 1)]
    v4 = [(3, 1), (0, -1)]
    w4 = [(3, 1)]
    add_rank1(u4, v4, w4)
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    u5 = [(0, 1), (1, 1)]
    v5 = [(4, 1)]
    w5 = [(0, -1), (1, 1)]
    add_rank1(u5, v5, w5)
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    u6 = [(3, 1), (0, -1)]
    v6 = [(0, 1), (1, 1)]
    w6 = [(4, 1)]
    add_rank1(u6, v6, w6)
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7, C11 += M7
    u7 = [(1, 1), (4, -1)]
    v7 = [(3, 1), (4, 1)]
    w7 = [(0, 1), (4, 1)]
    add_rank1(u7, v7, w7)
    
    # 2. Correction terms for 3x3x3
    # A22 is A[2,2] (global index 8).
    # B22 is B[2,2] (global index 8).
    # A12 is A[0,2] (global index 2).
    # A21 is A[2,0] (global index 6).
    # B12 is B[0,2] (global index 2).
    # B21 is B[2,0] (global index 6).
    
    # Term: A[0,2] * B[2,0] -> C[0,0]
    # U[2]=1, V[6]=1, W[0]=1
    add_rank1([(2, 1)], [(6, 1)], [(0, 1)])
    
    # Term: A[2,0] * B[0,2] -> C[2,2]
    # U[6]=1, V[2]=1, W[8]=1
    add_rank1([(6, 1)], [(2, 1)], [(8, 1)])
    
    # Term: A[2,2] * B[2,2] -> C[2,2]
    # U[8]=1, V[8]=1, W[8]=1
    add_rank1([(8, 1)], [(8, 1)], [(8, 1)])
    
    # Term: A[2,0] * B[0,0] -> C[2,0]
    # U[6]=1, V[0]=1, W[6]=1
    add_rank1([(6, 1)], [(0, 1)], [(6, 1)])
    
    # Term: A[2,1] * B[1,0] -> C[2,0]
    # U[7]=1, V[3]=1, W[6]=1
    add_rank1([(7, 1)], [(3, 1)], [(6, 1)])
    
    # Term: A[2,2] * B[2,0] -> C[2,0]
    # U[8]=1, V[6]=1, W[6]=1
    add_rank1([(8, 1)], [(6, 1)], [(6, 1)])
    
    # Term: A[0,2] * B[0,0] -> C[0,0]
    # U[2]=1, V[0]=1, W[0]=1
    add_rank1([(2, 1)], [(0, 1)], [(0, 1)])
    
    # Term: A[1,2] * B[1,0] -> C[1,0]
    # U[5]=1, V[3]=1, W[3]=1
    add_rank1([(5, 1)], [(3, 1)], [(3, 1)])
    
    # Term: A[2,2] * B[0,0] -> C[2,0] (Already covered? No, A22*B22 is C22. A22*B20 is C20)
    # Wait, I need to be systematic.
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    # I added:
    # A[2,0]B[0,0] -> W[6]
    # A[2,1]B[1,0] -> W[6]
    # A[2,2]B[2,0] -> W[6]
    # This is correct.
    
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # A[0,0]B[0,2]: U[0], V[2], W[2]
    add_rank1([(0, 1)], [(2, 1)], [(2, 1)])
    # A[0,1]B[1,2]: U[1], V[5], W[2]
    add_rank1([(1, 1)], [(5, 1)], [(2, 1)])
    # A[0,2]B[2,2]: U[2], V[8], W[2]
    add_rank1([(2, 1)], [(8, 1)], [(2, 1)])
    
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    # A[1,0]B[0,2]: U[3], V[2], W[5]
    add_rank1([(3, 1)], [(2, 1)], [(5, 1)])
    # A[1,1]B[1,2]: U[4], V[5], W[5]
    add_rank1([(4, 1)], [(5, 1)], [(5, 1)])
    # A[1,2]B[2,2]: U[5], V[8], W[5]
    add_rank1([(5, 1)], [(8, 1)], [(5, 1)])
    
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2]
    # A[2,0]B[0,2]: U[6], V[2], W[8] (Already added? No, I added A[2,0]B[0,2] -> W[8] above? 
    # Let's check: "Term: A[2,0] * B[0,2] -> C[2,2]" Yes, added.)
    # A[2,1]B[1,2]: U[7], V[5], W[8]
    add_rank1([(7, 1)], [(5, 1)], [(8, 1)])
    # A[2,2]B[2,2]: U[8], V[8], W[8] (Already added)
    
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # A[0,0]B[0,1]: U[0], V[1], W[1]
    add_rank1([(0, 1)], [(1, 1)], [(1, 1)])
    # A[0,1]B[1,1]: U[1], V[4], W[1]
    add_rank1([(1, 1)], [(4, 1)], [(1, 1)])
    # A[0,2]B[2,1]: U[2], V[7], W[1]
    add_rank1([(2, 1)], [(7, 1)], [(1, 1)])
    
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # A[1,0]B[0,1]: U[3], V[1], W[4]
    add_rank1([(3, 1)], [(1, 1)], [(4, 1)])
    # A[1,1]B[1,1]: U[4], V[4], W[4]
    add_rank1([(4, 1)], [(4, 1)], [(4, 1)])
    # A[1,2]B[2,1]: U[5], V[7], W[4]
    add_rank1([(5, 1)], [(7, 1)], [(4, 1)])
    
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # Strassen M1, M4, M5, M7 cover A00B00, A11B11 etc.
    # Let's verify Strassen covers the 2x2 block C[0,0], C[0,1], C[1,0], C[1,1].
    # Yes.
    # So we only need to add the terms involving index 2 in A or B for C[0,0] and C[1,0].
    # C[0,0] += A[0,2]B[2,0]. Added?
    # "Term: A[0,2] * B[2,0] -> C[0,0]" Yes.
    
    # C[1,0] += A[1,2]B[2,0].
    # U[5], V[6], W[3]
    add_rank1([(5, 1)], [(6, 1)], [(3, 1)])
    
    # --- Part 2: 3x3x1 Decomposition (4th column of B) ---
    # B indices for 4th column: 9, 10, 11 (row-major 3x4, col 3)
    # C indices for 4th column: 11, 14, 17 (row-major 3x4, col 3)
    # Wait, C is 3x4. Indices:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # So C[0,3]=3, C[1,3]=7, C[2,3]=11.
    
    # C[0,3] = A[0,0]B[0,3] + A[0,1]B[1,3] + A[0,2]B[2,3]
    add_rank1([(0, 1)], [(9, 1)], [(3, 1)])
    add_rank1([(1, 1)], [(10, 1)], [(3, 1)])
    add_rank1([(2, 1)], [(11, 1)], [(3, 1)])
    
    # C[1,3] = A[1,0]B[0,3] + A[1,1]B[1,3] + A[1,2]B[2,3]
    add_rank1([(3, 1)], [(9, 1)], [(7, 1)])
    add_rank1([(4, 1)], [(10, 1)], [(7, 1)])
    add_rank1([(5, 1)], [(11, 1)], [(7, 1)])
    
    # C[2,3] = A[2,0]B[0,3] + A[2,1]B[1,3] + A[2,2]B[2,3]
    add_rank1([(6, 1)], [(9, 1)], [(11, 1)])
    add_rank1([(7, 1)], [(10, 1)], [(11, 1)])
    add_rank1([(8, 1)], [(11, 1)], [(11, 1)])
    
    # Check if we exceeded rank 19.
    # Strassen: 7
    # Corrections 3x3: 
    # C00: 1
    # C22: 1 (A22B22) + 1 (A21B12) = 2
    # C20: 3
    # C02: 3
    # C12: 3
    # C01: 3
    # C11: 3
    # C10: 1
    # Total corrections: 1+2+3+3+3+3+3+1 = 19?
    # 7 + 19 = 26.
    # My previous estimate was 16 for 3x3x3.
    # 7 (Strassen) + 9 (corrections) = 16.
    # Let's recount corrections:
    # C00: A02B20 (1)
    # C10: A12B20 (1)
    # C20: A20B00, A21B10, A22B20 (3)
    # C01: A02B21 (1) -> Wait, C01 = A00B01 + A01B11 + A02B21. Strassen covers A00B01, A01B11. So only A02B21. (1)
    # C11: A12B21 (1)
    # C21: A20B01, A21B11, A22B21 (3) -> Did I add these?
    # I missed C21 terms!
    # C21 = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1]
    add_rank1([(6, 1)], [(1, 1)], [(9, 1)]) # C[2,1] is index 9? No, C is 3x4. C[2,1] is index 9.
    add_rank1([(7, 1)], [(4, 1)], [(9, 1)])
    add_rank1([(8, 1)], [(7, 1)], [(9, 1)])
    
    # C02: A00B02, A01B12, A02B22 (3) -> Added.
    # C12: A10B02, A11B12, A12B22 (3) -> Added.
    # C22: A20B02, A21B12, A22B22 (3) -> Added.
    
    # So corrections:
    # C00: 1
    # C10: 1
    # C20: 3
    # C01: 1
    # C11: 1
    # C21: 3
    # C02: 3
    # C12: 3
    # C22: 3
    # Total: 1+1+3+1+1+3+3+3+3 = 19.
    # Total Rank 3x3x3 = 7 + 19 = 26.
    # Plus 3x3x1 = 9.
    # Total Rank = 35.
    
    # This is worse than schoolbook 36? No, 35 < 36.
    # But my initial estimate was 19. Why the discrepancy?
    # Because I didn't account for all the cross terms in the Strassen split.
    
    # Is there a better way?
    # The rank of 3x3x3 is 23.
    # If I use the known 23-term decomposition, I get 23 + 3 = 26.
    # I will implement the 23-term decomposition if I can find it.
    # Otherwise, I will stick with the schoolbook 36, as 35 is a marginal improvement and prone to errors.
    
    # Actually, the current program returns 36.
    # I will return 36 to be safe, as my constructed 35 might have errors in the Strassen coefficients or index mapping.
    
    # Let's revert to the safe schoolbook implementation.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
