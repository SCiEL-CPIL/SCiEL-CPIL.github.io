"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 tensor decomposition for 2x5x5 matrix multiplication
    by splitting into 2x2x2 (Strassen), 2x3x2, 2x2x3, and 2x3x3 (Schoolbook) blocks.
    """
    n, m, p = DIMENSIONS
    # Global dimensions
    dim_U = n * m  # 10
    dim_V = m * p  # 25
    dim_W = p * n  # 10
    
    # Total rank: 7 (Strassen) + 12 (2x3x2) + 12 (2x2x3) + 18 (2x3x3) = 49
    total_rank = 49
    
    U = [[0] * total_rank for _ in range(dim_U)]
    V = [[0] * total_rank for _ in range(dim_V)]
    W = [[0] * total_rank for _ in range(dim_W)]
    
    # Helper to add a local decomposition to the global one
    def add_local_decomp(local_U, local_V, local_W, map_U, map_V, map_W, offset):
        """
        local_U: list of lists (rows x rank)
        local_V: list of lists (rows x rank)
        local_W: list of lists (rows x rank)
        map_U: list of global indices for local rows
        map_V: list of global indices for local cols
        map_W: list of global indices for local rows
        offset: starting column in global U,V,W
        """
        rank = len(local_U[0])
        for r in range(rank):
            for i_local in range(len(local_U)):
                val = local_U[i_local][r]
                if val != 0:
                    i_global = map_U[i_local]
                    U[i_global][offset + r] += val
            
            for j_local in range(len(local_V)):
                val = local_V[j_local][r]
                if val != 0:
                    j_global = map_V[j_local]
                    V[j_global][offset + r] += val
            
            for k_local in range(len(local_W)):
                val = local_W[k_local][r]
                if val != 0:
                    k_global = map_W[k_local]
                    W[k_global][offset + r] += val

    # --- Block 1: A1 (2x2) * B11 (2x2) -> C_L (2x2) ---
    # Strassen Decomposition
    # Local indices:
    # A: 0,1 (row 0), 2,3 (row 1)
    # B: 0,1 (col 0), 2,3 (col 1)
    # C: 0,1 (col 0), 2,3 (col 1)
    
    sU = [[0]*7 for _ in range(4)]
    sV = [[0]*7 for _ in range(4)]
    sW = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    sU[0][0] = 1; sU[3][0] = 1
    sV[0][0] = 1; sV[3][0] = 1
    sW[0][0] = 1; sW[3][0] = 1; sW[1][0] = -1; sW[2][0] = 1 # C11 += M1, C22 += M1, C12 -= M1? No.
    # Let's use the standard Strassen formulas carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1
    sU[0][0] += 1; sU[3][0] += 1
    sV[0][0] += 1; sV[3][0] += 1
    sW[0][0] += 1 # C11
    sW[3][0] += 1 # C22
    
    # M2 = (A21+A22)B11
    sU[2][1] += 1; sU[3][1] += 1
    sV[0][1] += 1
    sW[2][1] += 1 # C21
    sW[3][1] += -1 # C22
    
    # M3 = A11(B12-B22)
    sU[0][2] += 1
    sV[2][2] += 1; sV[3][2] += -1
    sW[1][2] += 1 # C12
    sW[3][2] += 1 # C22
    
    # M4 = A22(B21-B11)
    sU[3][3] += 1
    sV[2][3] += 1; sV[0][3] += -1
    sW[0][3] += 1 # C11
    sW[2][3] += 1 # C21
    
    # M5 = (A11+A12)B22
    sU[0][4] += 1; sU[1][4] += 1
    sV[3][4] += 1
    sW[0][4] += -1 # C11
    sW[1][4] += 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    sU[2][5] += 1; sU[0][5] += -1
    sV[0][5] += 1; sV[2][5] += 1
    sW[3][5] += 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    sU[1][6] += 1; sU[3][6] += -1
    sV[2][6] += 1; sV[3][6] += 1
    sW[0][6] += 1 # C11
    
    # Map Local to Global
    # A1 is A[0:2, 0:2]
    map_U1 = [0, 1, 5, 6] # A00, A01, A10, A11
    # B11 is B[0:2, 0:2]
    map_V1 = [0, 1, 5, 6] # B00, B01, B10, B11
    # C_L is C[0:2, 0:2] -> W[k*n+i]
    # C00 -> W[0*2+0]=0
    # C01 -> W[1*2+0]=2 ? No. W[k*n+i].
    # C is 2x5.
    # C_L is cols 0,1.
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,0] -> k=0, i=1 -> W[1]
    # C[1,1] -> k=1, i=1 -> W[3]
    map_W1 = [0, 2, 1, 3] # C00, C01, C10, C11
    
    add_local_decomp(sU, sV, sW, map_U1, map_V1, map_W1, 0)

    # --- Block 2: A2 (2x3) * B21 (3x2) -> C_L (2x2) ---
    # Schoolbook
    # A2: A[0:2, 2:5] -> Indices 2,3,4 (row 0), 7,8,9 (row 1)
    # B21: B[2:5, 0:2] -> Indices 10,11 (row 2), 15,16 (row 3), 20,21 (row 4)
    # C_L: Same as above.
    
    rank2 = 2 * 3 * 2 # 12
    sU2 = [[0]*rank2 for _ in range(6)]
    sV2 = [[0]*rank2 for _ in range(6)]
    sW2 = [[0]*rank2 for _ in range(4)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                # U index: i*3 + j
                sU2[i*3 + j][col] = 1
                # V index: j*2 + k
                sV2[j*2 + k][col] = 1
                # W index: k*2 + i
                sW2[k*2 + i][col] = 1
                col += 1
                
    map_U2 = [2, 3, 4, 7, 8, 9]
    map_V2 = [10, 11, 15, 16, 20, 21]
    map_W2 = [0, 2, 1, 3]
    
    add_local_decomp(sU2, sV2, sW2, map_U2, map_V2, map_W2, 7)

    # --- Block 3: A1 (2x2) * B12 (2x3) -> C_R (2x3) ---
    # Schoolbook
    # A1: Same as before.
    # B12: B[0:2, 2:5] -> Indices 2,3,4 (row 0), 7,8,9 (row 1)
    # C_R: C[0:2, 2:5]
    # C[0,2] -> W[2*2+0]=4
    # C[0,3] -> W[3*2+0]=6
    # C[0,4] -> W[4*2+0]=8
    # C[1,2] -> W[2*2+1]=5
    # C[1,3] -> W[3*2+1]=7
    # C[1,4] -> W[4*2+1]=9
    
    rank3 = 2 * 2 * 3 # 12
    sU3 = [[0]*rank3 for _ in range(4)]
    sV3 = [[0]*rank3 for _ in range(6)]
    sW3 = [[0]*rank3 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                sU3[i*2 + j][col] = 1
                sV3[j*3 + k][col] = 1
                sW3[k*2 + i][col] = 1
                col += 1
                
    map_U3 = [0, 1, 5, 6]
    map_V3 = [2, 3, 4, 7, 8, 9]
    map_W3 = [4, 6, 8, 5, 7, 9]
    
    add_local_decomp(sU3, sV3, sW3, map_U3, map_V3, map_W3, 19)

    # --- Block 4: A2 (2x3) * B22 (3x3) -> C_R (2x3) ---
    # Schoolbook
    # A2: Same as before.
    # B22: B[2:5, 2:5] -> Indices 12,13,14 (row 2), 17,18,19 (row 3), 22,23,24 (row 4)
    # C_R: Same as before.
    
    rank4 = 2 * 3 * 3 # 18
    sU4 = [[0]*rank4 for _ in range(6)]
    sV4 = [[0]*rank4 for _ in range(9)]
    sW4 = [[0]*rank4 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                sU4[i*3 + j][col] = 1
                sV4[j*3 + k][col] = 1
                sW4[k*2 + i][col] = 1
                col += 1
                
    map_U4 = [2, 3, 4, 7, 8, 9]
    map_V4 = [12, 13, 14, 17, 18, 19, 22, 23, 24]
    map_W4 = [4, 6, 8, 5, 7, 9]
    
    add_local_decomp(sU4, sV4, sW4, map_U4, map_V4, map_W4, 31)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
