"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 1: 4x4x4 Strassen (49)
    # Rank 2: 4x4x3 Schoolbook (48)
    rank1 = 49
    rank2 = 48
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen ---
    # 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: A00+A11, B00+B11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00 += M1, C11 += M1
    
    # M2: A10+A11, B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1 # C10 += M2, C11 -= M2
    
    # M3: A00, B10-B11
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01 += M3, C11 += M3
    
    # M4: A11, B01-B00
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00 += M4, C10 += M4
    
    # M5: A00+A01, B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00 -= M5, C01 += M5
    
    # M6: A10-A00, B00+B01
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11 += M6
    
    # M7: A01-A11, B10+B11
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00 += M7
    
    # Block coefficients for 4x4 Strassen
    # A_coeffs[K][I][J]
    A_coeffs = [
        [[1,0],[0,1]], # M1: A00+A11
        [[0,0],[1,1]], # M2: A10+A11
        [[1,0],[0,0]], # M3: A00
        [[0,0],[0,1]], # M4: A11
        [[1,1],[0,0]], # M5: A00+A01
        [[-1,0],[1,0]],# M6: A10-A00
        [[0,1],[0,-1]] # M7: A01-A11
    ]
    B_coeffs = [
        [[1,0],[0,1]], # M1: B00+B11
        [[1,0],[0,0]], # M2: B00
        [[0,0],[1,-1]],# M3: B10-B11
        [[-1,1],[0,0]],# M4: B01-B00
        [[0,0],[0,1]], # M5: B11
        [[1,1],[0,0]], # M6: B00+B01
        [[0,0],[1,1]]  # M7: B10+B11
    ]
    C_coeffs = [
        [1, 0], # C00 gets M1
        [0, 0], # C01 gets none from M2? No, C01 gets M3+M5
        [0, 0], # C10 gets M2+M4
        [1, 0]  # C11 gets M1
    ]
    # Correct C_coeffs:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0], # C00: M1
        [0, 0], # C01: 0
        [0, 0], # C10: 0
        [1, 0]  # C11: M1
    ]
    C_coeffs[0][0] = 1; C_coeffs[3][0] = 1
    C_coeffs[0][1] = 1; C_coeffs[2][1] = 1
    C_coeffs[0][2] = -1; C_coeffs[1][2] = 1
    C_coeffs[0][3] = 1; C_coeffs[2][3] = 1
    C_coeffs[1][4] = 1; C_coeffs[0][4] = -1
    C_coeffs[3][5] = 1
    C_coeffs[0][6] = 1
    
    # Wait, C_coeffs is indexed by [BlockI, BlockJ] for output?
    # No, C_coeffs[BlockI, BlockJ] is a vector of length 7 (for M1..M7).
    # Let's define C_coeffs as 2x2x7
    C_coeffs = [[[0]*7 for _ in range(2)] for _ in range(2)]
    # C00 (Block 0,0)
    C_coeffs[0][0][0] = 1 # M1
    C_coeffs[0][0][3] = 1 # M4
    C_coeffs[0][0][4] = -1 # M5
    C_coeffs[0][0][6] = 1 # M7
    # C01 (Block 0,1)
    C_coeffs[0][1][2] = 1 # M3
    C_coeffs[0][1][4] = 1 # M5
    # C10 (Block 1,0)
    C_coeffs[1][0][1] = 1 # M2
    C_coeffs[1][0][3] = 1 # M4
    # C11 (Block 1,1)
    C_coeffs[1][1][0] = 1 # M1
    C_coeffs[1][1][1] = -1 # M2
    C_coeffs[1][1][2] = 1 # M3
    C_coeffs[1][1][5] = 1 # M6
    
    # Fill U, V, W for 4x4x4
    for K in range(7):
        for s in range(7):
            col = K * 7 + s
            
            # U
            for I in range(2):
                for J in range(2):
                    coeff_A = A_coeffs[K][I][J]
                    if coeff_A != 0:
                        for i_loc in range(2):
                            for j_loc in range(2):
                                r2 = i_loc * 2 + j_loc
                                val = U2[r2][s]
                                if val != 0:
                                    i = I * 2 + i_loc
                                    j = J * 2 + j_loc
                                    U[i * 4 + j][col] += coeff_A * val
                                    
            # V
            for I in range(2):
                for J in range(2):
                    coeff_B = B_coeffs[K][I][J]
                    if coeff_B != 0:
                        for i_loc in range(2):
                            for j_loc in range(2):
                                r2 = i_loc * 2 + j_loc
                                val = V2[r2][s]
                                if val != 0:
                                    i = I * 2 + i_loc
                                    j = J * 2 + j_loc
                                    V[i * 4 + j][col] += coeff_B * val
                                    
            # W
            for I in range(2):
                for J in range(2):
                    coeff_C = C_coeffs[I][J][K]
                    if coeff_C != 0:
                        for i_loc in range(2):
                            for j_loc in range(2):
                                k2 = i_loc * 2 + j_loc
                                val = W2[k2][s]
                                if val != 0:
                                    i = I * 2 + i_loc
                                    j = J * 2 + j_loc
                                    W[j * 4 + i][col] += coeff_C * val
                                    
    # --- Part 2: 4x4x3 Schoolbook ---
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k_global = 4 + k_local
                col = col_offset + (i * 4 + j) * 3 + k_local
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
