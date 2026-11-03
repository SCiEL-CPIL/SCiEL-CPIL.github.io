"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2x2 blocks
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Define Strassen 2x2x2 factors
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U_S[0][0] = 1; U_S[3][0] = 1
    V_S[0][0] = 1; V_S[3][0] = 1
    W_S[2][0] = 1 # C00
    W_S[1][0] = 1 # C11
    
    # M2: (A10+A11)B00
    U_S[2][1] = 1; U_S[3][1] = 1
    V_S[0][1] = 1
    W_S[0][1] = 1 # C10
    W_S[1][1] = -1 # C11
    
    # M3: A00(B01-B11)
    U_S[0][2] = 1
    V_S[1][2] = 1; V_S[3][2] = -1
    W_S[3][2] = 1 # C01
    W_S[1][2] = 1 # C11
    
    # M4: A11(B10-B00)
    U_S[3][3] = 1
    V_S[2][3] = 1; V_S[0][3] = -1
    W_S[2][3] = 1 # C00
    W_S[0][3] = 1 # C10
    
    # M5: (A00+A01)B11
    U_S[0][4] = 1; U_S[1][4] = 1
    V_S[3][4] = 1
    W_S[2][4] = -1 # C00
    W_S[3][4] = 1 # C01
    
    # M6: (A10-A00)(B00+B01)
    U_S[2][5] = 1; U_S[0][5] = -1
    V_S[0][5] = 1; V_S[1][5] = 1
    W_S[1][5] = 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U_S[1][6] = 1; U_S[3][6] = -1
    V_S[2][6] = 1; V_S[3][6] = 1
    W_S[2][6] = 1 # C00

    # Build 2x2x4 factors (Rank 14)
    # U_224: 4x14
    # V_224: 8x14
    # W_224: 8x14
    U_224 = [[0]*14 for _ in range(4)]
    V_224 = [[0]*14 for _ in range(8)]
    W_224 = [[0]*14 for _ in range(8)]
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                idx_u = 2*i + j
                U_224[idx_u][c] = U_S[idx_u][c]
                U_224[idx_u][c+7] = U_S[idx_u][c]
                
        for j in range(2):
            for k in range(2):
                idx_v = 2*j + k
                V_224[idx_v][c] = V_S[idx_v][c]
                # For c+7, k maps to k+2
                V_224[2*j + (k+2)][c+7] = V_S[idx_v][c]
                
        for k in range(2):
            for i in range(2):
                idx_w = 2*k + i
                W_224[idx_w][c] = W_S[idx_w][c]
                # For c+7, k maps to k+2
                W_224[2*(k+2) + i][c+7] = W_S[idx_w][c]

    # Assemble global 2x4x4 factors (Rank 28)
    # Part 1: A_L (cols 0,1) * B_L (rows 0,1) -> C
    # Part 2: A_R (cols 2,3) * B_R (rows 2,3) -> C
    
    for c in range(14):
        # U: 8x28
        # Global U index: 4*i + j
        # Part 1 uses j=0,1
        for i in range(2):
            for j in range(2):
                U[4*i + j][c] = U_224[2*i + j][c]
        
        # Part 2 uses j=2,3 (offset by 7 in c)
        for i in range(2):
            for j in range(2):
                # Global j = 2 + j
                U[4*i + (2+j)][c+14] = U_224[2*i + j][c]

        # V: 16x28
        # Global V index: 4*j + k
        # Part 1 uses j=0,1
        for j in range(2):
            for k in range(4):
                # V_224 index: 2*j + k
                V[4*j + k][c] = V_224[2*j + k][c]
        
        # Part 2 uses j=2,3 (offset by 14 in c)
        for j in range(2):
            for k in range(4):
                # Global j = 2 + j
                V[4*(2+j) + k][c+14] = V_224[2*j + k][c]

        # W: 8x28
        # Global W index: 2*k + i
        # Both parts contribute to all k, i
        for k in range(4):
            for i in range(2):
                W[2*k + i][c] = W_224[2*k + i][c]
                W[2*k + i][c+14] = W_224[2*k + i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
