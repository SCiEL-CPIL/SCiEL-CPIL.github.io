"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrices.
    """
    n, m, p = DIMENSIONS
    
    # Base Strassen decomposition for 2x2 matrices
    # U2: (i, j) -> index i*2 + j
    # V2: (j, k) -> index j*2 + k
    # W2: (k, i) -> index k*2 + i
    # Rank 7
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: A00+A11, B00+B11 -> C00+ C11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00 (k=0,i=0), C11 (k=1,i=1)
    
    # M2: A10+A11, B00 -> C10 - C11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1 # C10 (k=0,i=1), C11 (k=1,i=1)
    
    # M3: A00, B01-B11 -> C01 + C11
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1 # C01 (k=1,i=0), C11 (k=1,i=1)
    
    # M4: A11, B10-B00 -> C00 + C10
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1 # C00 (k=0,i=0), C10 (k=0,i=1)
    
    # M5: A00+A10, B11 -> -C00 + C01
    U2[0][4] = 1; U2[2][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1 # C00 (k=0,i=0), C01 (k=1,i=0)
    
    # M6: A01-A00, B00+B10 -> C11
    U2[1][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1 # C11 (k=1,i=1)
    
    # M7: A10-A11, B11+B01 -> C00
    U2[2][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[1][6] = 1
    W2[0][6] = 1 # C00 (k=0,i=0)
    
    # Construct 4x4 decomposition
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            
            # Fill U
            for u in range(n*m):
                i = u // m
                j = u % m
                i1, i2 = i // 2, i % 2
                j1, j2 = j // 2, j % 2
                rA = i1 * 2 + j1
                rB = i2 * 2 + j2
                U[u][c] = U2[rA][c1] * U2[rB][c2]
                
            # Fill V
            for v in range(m*p):
                j = v // p
                k = v % p
                j1, j2 = j // 2, j % 2
                k1, k2 = k // 2, k % 2
                rA = j1 * 2 + k1
                rB = j2 * 2 + k2
                V[v][c] = V2[rA][c1] * V2[rB][c2]
                
            # Fill W
            for w in range(p*n):
                k = w // n
                i = w % n
                k1, k2 = k // 2, k % 2
                i1, i2 = i // 2, i % 2
                rA = k1 * 2 + i1
                rB = k2 * 2 + i2
                W[w][c] = W2[rA][c1] * W2[rB][c2]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
