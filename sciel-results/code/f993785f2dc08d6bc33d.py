"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two Strassen blocks
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define Strassen coefficients for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    # W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2), 6:(0,3), 7:(1,3)
    
    # Block 1: B cols 0,1 -> C cols 0,1
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 3: M4
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5: M6
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7
    U[2][6] = 1; U[0][6] = -1
    V[5][6] = 1; V[4][6] = 1
    W[0][6] = 1

    # Block 2: B cols 2,3 -> C cols 2,3
    # V indices shifted by +2
    # W indices shifted by +4
    
    # Term 7: M1
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # Term 8: M2
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # Term 9: M3
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # Term 10: M4
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # Term 11: M5
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # Term 12: M6
    U[1][12] = 1; U[3][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # Term 13: M7
    U[2][13] = 1; U[0][13] = -1
    V[7][13] = 1; V[6][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
