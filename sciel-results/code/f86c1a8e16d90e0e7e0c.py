"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: 7 (Strassen for 2x2x2) + 4 (Schoolbook for 2x2x1)
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i*m + j (4 entries)
    # V: j*p + k (6 entries)
    # W: k*n + i (6 entries)
    
    # Strassen terms for B cols 0,1
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4
    U[3][3] = 1
    V[0][3] = -1; V[3][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Schoolbook terms for B col 2
    # A00 B02 -> C02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # A01 B12 -> C02
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # A10 B02 -> C12
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # A11 B12 -> C12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
