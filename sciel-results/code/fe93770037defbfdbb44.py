"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize U, V, W as lists of lists
    # U: n*m = 4 rows, rank cols
    # V: m*p = 4 rows, rank cols
    # W: p*n = 4 rows, rank cols
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's Algorithm Decomposition
    # Indices for U (A input): 00, 01, 10, 11
    # Indices for V (B input): 00, 01, 10, 11
    # Indices for W (C output): 00, 01, 10, 11
    
    # r=1: C00 += (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=2: C10 += (A10+A11)(B00)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # r=3: C01 += (A00)(B10+B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = 1
    W[1][2] = 1
    
    # r=4: C11 += (A11)(B11-B00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # r=5: C00 += (A00+A01)(B11)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    # r=6: C11 -= (A10+A00)(B01)
    U[2][5] = 1; U[0][5] = 1
    V[1][5] = 1
    W[3][5] = -1
    
    # r=7: C10 -= (A01-A10)(B00+B10)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[2][6] = -1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
