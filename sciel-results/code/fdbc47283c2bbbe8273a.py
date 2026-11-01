"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition
    rank = 11
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (U): 0:A00, 1:A01, 2:A10, 3:A11
    # Indices for B (V): 0:B00, 1:B01, 2:B02, 3:B10, 4:B11, 5:B12
    # Indices for C (W): 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12
    
    # Part 1: Strassen for C00, C10, C01, C11 (using B00, B01, B10, B11)
    # r=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1
    U[2][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # r=4
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=5
    U[0][5] = 1; U[2][5] = 1
    V[1][5] = 1; V[4][5] = -1
    W[3][5] = 1
    
    # r=6
    U[1][6] = 1
    V[4][6] = 1
    W[0][6] = 1
    
    # Part 2: Schoolbook for C02, C12 (using B02, B12)
    # r=7
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # r=8
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # r=9
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # r=10
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
