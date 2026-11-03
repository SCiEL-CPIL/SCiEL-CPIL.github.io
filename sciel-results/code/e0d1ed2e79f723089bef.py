"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-17 decomposition: Strassen (7) + schoolbook (10)
    # C = A*B, A is 2x3, B is 3x3
    # W convention: W[k*n+i, r] where n=2
    # U: 6x17, V: 9x17, W: 6x17
    rank = 17
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(6)]
    
    # Strassen decomposition for A1*B_L (A1 = A[:,0:2], B_L = B[0:2,:])
    # 7 products:
    # P1 = (A1_00 + A1_11) * (B_L_00 + B_L_11)
    # P2 = (A1_00 + A1_10) * B_L_11
    # P3 = A1_00 * (B_L_01 - B_L_11)
    # P4 = A1_11 * (B_L_10 - B_L_00)
    # P5 = (A1_00 + A1_01) * B_L_00
    # P6 = (A1_10 - A1_00) * (B_L_00 + B_L_01)
    # P7 = (A1_01 - A1_11) * (B_L_10 + B_L_11)
    #
    # C = P1 - P3 - P4 + P7  (diagonal terms)
    #    + P2 + P5           (row terms)
    #    + P6 + P7           (column terms)
    #
    # Specifically:
    # C00 = P1 + P2 - P3 - P4
    # C01 = P5 - P7
    # C10 = P3 + P4 - P6
    # C11 = P1 - P2 + P6 + P7
    
    # Product 0: P1 = (A_00 + A_11) * (B_00 + B_11)
    U[0][0] = 1  # A_00
    U[3][0] = 1  # A_11
    V[0][0] = 1  # B_00
    V[3][0] = 1  # B_11
    W[0][0] = 1  # C_00
    W[3][0] = 1  # C_11
    
    # Product 1: P2 = (A_00 + A_10) * B_11
    U[0][1] = 1  # A_00
    U[2][1] = 1  # A_10
    V[3][1] = 1  # B_11
    W[0][1] = 1  # C_00
    W[3][1] = -1 # C_11
    
    # Product 2: P3 = A_00 * (B_01 - B_11)
    U[0][2] = 1  # A_00
    V[1][2] = 1  # B_01
    V[3][2] = -1 # B_11
    W[0][2] = -1 # C_00
    W[2][2] = 1  # C_10
    
    # Product 3: P4 = A_11 * (B_10 - B_00)
    U[3][3] = 1  # A_11
    V[2][3] = 1  # B_10
    V[0][3] = -1 # B_00
    W[0][3] = -1 # C_00
    W[2][3] = 1  # C_10
    
    # Product 4: P5 = (A_00 + A_01) * B_00
    U[0][4] = 1  # A_00
    U[1][4] = 1  # A_01
    V[0][4] = 1  # B_00
    W[0][4] = 1  # C_01
    
    # Product 5: P6 = (A_10 - A_00) * (B_00 + B_01)
    U[2][5] = 1  # A_10
    U[0][5] = -1 # A_00
    V[0][5] = 1  # B_00
    V[1][5] = 1  # B_01
    W[2][5] = -1 # C_10
    W[3][5] = 1  # C_11
    
    # Product 6: P7 = (A_01 - A_11) * (B_10 + B_11)
    U[1][6] = 1  # A_01
    U[3][6] = -1 # A_11
    V[2][6] = 1  # B_10
    V[3][6] = 1  # B_11
    W[0][6] = -1 # C_01
    W[3][6] = 1  # C_11
    
    # Schoolbook for A1 * B_R (A1 = A[:,0:2], B_R = B[2,:])
    # 4 products
    # Product 7: A_00 * B_20
    U[0][7] = 1  # A_00
    V[4][7] = 1  # B_20
    W[4][7] = 1  # C_02
    
    # Product 8: A_01 * B_20
    U[1][8] = 1  # A_01
    V[4][8] = 1  # B_20
    W[2][8] = 1  # C_01
    
    # Product 9: A_10 * B_20
    U[2][9] = 1  # A_10
    V[4][9] = 1  # B_20
    W[1][9] = 1  # C_10
    
    # Product 10: A_11 * B_20
    U[3][10] = 1  # A_11
    V[4][10] = 1  # B_20
    W[5][10] = 1  # C_12
    
    # Schoolbook for A2 * B (A2 = A[:,2], B is 3x3)
    # 6 products
    # Product 11: A_02 * B_00
    U[4][11] = 1  # A_02
    V[0][11] = 1  # B_00
    W[0][11] = 1  # C_00
    
    # Product 12: A_02 * B_01
    U[4][12] = 1  # A_02
    V[1][12] = 1  # B_01
    W[2][12] = 1  # C_01
    
    # Product 13: A_02 * B_02
    U[4][13] = 1  # A_02
    V[2][13] = 1  # B_02
    W[4][13] = 1  # C_02
    
    # Product 14: A_12 * B_00
    U[5][14] = 1  # A_12
    V[0][14] = 1  # B_00
    W[1][14] = 1  # C_10
    
    # Product 15: A_12 * B_01
    U[5][15] = 1  # A_12
    V[1][15] = 1  # B_01
    W[3][15] = 1  # C_11
    
    # Product 16: A_12 * B_02
    U[5][16] = 1  # A_12
    V[2][16] = 1  # B_02
    W[5][16] = 1  # C_12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

