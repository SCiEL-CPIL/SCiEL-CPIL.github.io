"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of 2x2 Strassen (Rank 7) and 3x3 Schoolbook (Rank 27)
    # Total Rank = 7 * 27 = 189
    
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    
    # Strassen 2x2 coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for Strassen blocks:
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # S_A[r][i_b][j_b]
    # S_B[r][j_b][k_b]
    # S_C[r][k_b][i_b]
    
    S_A = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [0, -1]]  # M7: A01 - A11
    ]
    
    S_B = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[0, -1], [1, 0]], # M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 0], [1, 1]]  # M7: B10 + B11
    ]
    
    S_C = [
        [[1, 0], [0, 1]], # M1: C00 + C11
        [[0, 0], [1, -1]], # M2: C10 - C11 (C10 gets +1, C11 gets -1)
        [[0, 1], [0, 1]], # M3: C01 + C11
        [[1, 0], [1, 0]], # M4: C00 + C10
        [[-1, 1], [0, 0]], # M5: -C00 + C01
        [[0, 0], [0, 1]], # M6: C11
        [[1, 0], [0, 0]]  # M7: C00
    ]
    
    # Laderman 3x3 coefficients (Schoolbook Rank 27)
    # We use Schoolbook for 3x3 to ensure correctness.
    # Rank = 27.
    
    # U_L, V_L, W_L for 3x3 Schoolbook
    # Index mapping: i in 0..2, j in 0..2, k in 0..2
    # U[i*3+j], V[j*3+k], W[k*3+i]
    
    U_L = [[0]*27 for _ in range(9)]
    V_L = [[0]*27 for _ in range(9)]
    W_L = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_L[i*3+j][col] = 1
                V_L[j*3+k][col] = 1
                W_L[k*3+i][col] = 1
                col += 1
                
    # Total Rank
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r in range(7):
        for l in range(27):
            idx = r * 27 + l
            
            # Fill U
            for i_b in range(2):
                for j_b in range(2):
                    coeff_A = S_A[r][i_b][j_b]
                    if coeff_A == 0: continue
                    for i_l in range(3):
                        for j_l in range(3):
                            coeff_L = U_L[i_l*3+j_l][l]
                            if coeff_L == 0: continue
                            I = 3*i_b + i_l
                            J = 3*j_b + j_l
                            U[I][idx] += coeff_A * coeff_L
                            
            # Fill V
            for j_b in range(2):
                for k_b in range(2):
                    coeff_B = S_B[r][j_b][k_b]
                    if coeff_B == 0: continue
                    for j_l in range(3):
                        for k_l in range(3):
                            coeff_L = V_L[j_l*3+k_l][l]
                            if coeff_L == 0: continue
                            J = 3*j_b + j_l
                            K = 3*k_b + k_l
                            V[J][idx] += coeff_B * coeff_L
                            
            # Fill W
            for k_b in range(2):
                for i_b in range(2):
                    coeff_C = S_C[r][k_b][i_b]
                    if coeff_C == 0: continue
                    for k_l in range(3):
                        for i_l in range(3):
                            coeff_L = W_L[k_l*3+i_l][l]
                            if coeff_L == 0: continue
                            K = 3*k_b + k_l
                            I = 3*i_b + i_l
                            W[K][idx] += coeff_C * coeff_L
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
