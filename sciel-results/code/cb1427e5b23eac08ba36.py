"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 5 blocks of 2 columns (Strassen, rank 7 each) and 1 block of 1 column (Trivial, rank 4)
    # Total rank = 5 * 7 + 4 = 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Process 5 blocks of 2 columns
    for block_idx in range(5):
        k1 = block_idx * 2
        k2 = block_idx * 2 + 1
        
        # Strassen's algorithm for 2x2 by 2x2
        # M1 = (a+d)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k2][col] = 1
        W[k1*n + 0][col] = 1; W[k2*n + 1][col] = 1
        col += 1
        
        # M2 = (c+d)B00
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1; W[k2*n + 1][col] = -1
        col += 1
        
        # M3 = a(B10+B11)
        U[0][col] = 1
        V[1*p + k1][col] = 1; V[1*p + k2][col] = 1
        W[k2*n + 0][col] = 1; W[k2*n + 1][col] = 1
        col += 1
        
        # M4 = d(B01-B11)
        U[3][col] = 1
        V[0*p + k2][col] = 1; V[1*p + k2][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M5 = (a+b)B11
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k2][col] = 1
        W[k2*n + 0][col] = 1; W[k1*n + 0][col] = -1
        col += 1
        
        # M6 = (c-a)(B00+B01)
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k1][col] = 1; V[0*p + k2][col] = 1
        W[k2*n + 1][col] = -1
        col += 1
        
        # M7 = (b-d)(B10+B11)
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k1][col] = 1; V[1*p + k2][col] = 1
        W[k1*n + 0][col] = 1
        col += 1

    # Process 1 block of 1 column (Trivial)
    k = 10
    # M1 = a B0
    U[0][col] = 1
    V[0*p + k][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    
    # M2 = b B1
    U[1][col] = 1
    V[1*p + k][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    
    # M3 = c B0
    U[2][col] = 1
    V[0*p + k][col] = 1
    W[k*n + 1][col] = 1
    col += 1
    
    # M4 = d B1
    U[3][col] = 1
    V[1*p + k][col] = 1
    W[k*n + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
