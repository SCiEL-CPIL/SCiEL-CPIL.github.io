DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Split B (4x6) into B1 (4x4) and B2 (4x2)
    # C = A @ B = A @ B1 + A @ B2
    # A @ B2 is 2x2. Rank 14.
    # A @ B1 is 2x4. Rank 20.
    # Total 34.
    
    # Part 1: A @ B2 (2x4 * 4x2 -> 2x2)
    # A = [A1, A2] (2x2), B2 = [B21; B22] (2x2)
    # C2 = A1 @ B21 + A2 @ B22
    # Use Strassen for each? No, just standard 7+7=14.
    
    col = 0
    # Strassen for A1 @ B21
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    # A1 is A[0:2, 0:2], B21 is B[0:2, 4:6]
    for s in range(7):
        for (i_loc, j_loc), coeff in U_coeffs[s].items():
            U[i_loc * m + j_loc][col] += coeff
        for (j_loc, k_loc), coeff in V_coeffs[s].items():
            V[j_loc * p + (4 + k_loc)][col] += coeff
        for (i_loc, k_loc), coeff in W_contrib[s].items():
            W[(4 + k_loc) * n + i_loc][col] += coeff
        col += 1
        
    # A2 is A[0:2, 2:4], B22 is B[2:4, 4:6]
    for s in range(7):
        for (i_loc, j_loc), coeff in U_coeffs[s].items():
            U[i_loc * m + (2 + j_loc)][col] += coeff
        for (j_loc, k_loc), coeff in V_coeffs[s].items():
            V[(2 + j_loc) * p + (4 + k_loc)][col] += coeff
        for (i_loc, k_loc), coeff in W_contrib[s].items():
            W[(4 + k_loc) * n + i_loc][col] += coeff
        col += 1
            
    # Part 2: A @ B1 (2x4 * 4x4 -> 2x4)
    # Split B1 into B11 (4x2) and B12 (4x2)
    # C1 = A @ B11 + A @ B12
    # Each is 2x2. Rank 14 each? No, 2x4 * 4x2 is 2x2.
    # A @ B11: A=[A1,A2], B11=[B111; B112]. C11 = A1@B111 + A2@B112. Rank 14.
    # A @ B12: A=[A1,A2], B12=[B121; B122]. C12 = A1@B121 + A2@B122. Rank 14.
    # Total 28.
    # Wait, 14+14+14 = 42.
    # My previous estimate was 34. Where did I go wrong?
    # M(2,4,4) = 28. M(2,4,2) = 14. Total 42.
    # So Rank 34 is not possible with this split.
    
    # I will return the Rank 42 solution as it is the best I can guarantee.
    # But I must improve.
    # I will use the Rank 40 solution if I can find the coefficients.
    # Since I can't, I will stick to 42 but maybe there is a Rank 36?
    # M(2,4,6) <= 36?
    
    # I will output the Rank 42 code.
    pass
