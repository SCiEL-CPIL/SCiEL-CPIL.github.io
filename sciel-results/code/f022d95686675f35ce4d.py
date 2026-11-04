"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block Strassen for 6x6 using 3x3 blocks
    # Rank = 7 * 27 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for A blocks in M_s
    # s: 0..6 corresponding to M_1..M_7
    # (a, b): block row a, block col b
    A_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1: A00 + A11
        {(1,0): 1, (1,1): 1},       # M2: A10 + A11
        {(0,0): 1},                 # M3: A00
        {(1,1): 1},                 # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00 + A01
        {(1,0): 1, (0,0): -1},      # M6: A10 - A00
        {(0,1): 1, (1,1): -1}       # M7: A01 - A11
    ]
    
    # Coefficients for B blocks in M_s
    B_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1: B00 + B11
        {(0,0): 1},                 # M2: B00
        {(0,1): 1, (1,1): -1},      # M3: B01 - B11
        {(1,0): 1, (0,0): -1},      # M4: B10 - B00
        {(1,1): 1},                 # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00 + B01
        {(1,0): 1, (1,1): 1}        # M7: B10 + B11
    ]
    
    # Coefficients for M_s in C blocks
    # C_ab = sum_s d_s_ab M_s
    C_coeffs = [
        {(0,0): 1, (1,1): -1, (0,1): 0, (1,0): 0}, # Placeholder, will fill below
        {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 0},
        {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 0},
        {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 0},
        {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 0},
        {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 0},
        {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 0}
    ]
    
    # Fill C_coeffs correctly
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # s=0 (M1): C00 += 1, C11 += 1
    C_coeffs[0][(0,0)] = 1
    C_coeffs[0][(1,1)] = 1
    # s=1 (M2): C10 += 1, C11 += -1
    C_coeffs[1][(1,0)] = 1
    C_coeffs[1][(1,1)] = -1
    # s=2 (M3): C01 += 1, C11 += 1
    C_coeffs[2][(0,1)] = 1
    C_coeffs[2][(1,1)] = 1
    # s=3 (M4): C00 += 1, C10 += 1
    C_coeffs[3][(0,0)] = 1
    C_coeffs[3][(1,0)] = 1
    # s=4 (M5): C00 += -1, C01 += 1
    C_coeffs[4][(0,0)] = -1
    C_coeffs[4][(0,1)] = 1
    # s=5 (M6): C11 += 1
    C_coeffs[5][(1,1)] = 1
    # s=6 (M7): C00 += 1
    C_coeffs[6][(0,0)] = 1

    # Loop over Strassen products s
    for s in range(7):
        # Loop over 3x3 schoolbook indices
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    r = s * 27 + i_prime * 9 + j_prime * 3 + k_prime
                    
                    # U part: A
                    for a in range(2):
                        for b in range(2):
                            val = A_coeffs[s].get((a,b), 0)
                            if val != 0:
                                i = 3 * a + i_prime
                                j = 3 * b + j_prime
                                U[i * m + j][r] = val
                                
                    # V part: B
                    for a in range(2):
                        for b in range(2):
                            val = B_coeffs[s].get((a,b), 0)
                            if val != 0:
                                j = 3 * a + j_prime
                                k = 3 * b + k_prime
                                V[j * p + k][r] = val
                                
                    # W part: C
                    for a in range(2):
                        for b in range(2):
                            val = C_coeffs[s].get((a,b), 0)
                            if val != 0:
                                i = 3 * a + i_prime
                                k = 3 * b + k_prime
                                W[k * n + i][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
