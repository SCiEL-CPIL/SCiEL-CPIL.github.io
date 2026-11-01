"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's Algorithm Rank 7 Decomposition
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # r=0: M1 = (a+d)(e+h)
    U[0][0] = 1; U[2][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # c00, c11
    
    # r=1: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1               # e
    W[1][1] = 1; W[3][1] = -1 # c10, c11
    
    # r=2: M3 = a(f-h)
    U[0][2] = 1               # a
    V[1][2] = 1; V[3][2] = -1 # f, -h
    W[2][2] = 1; W[3][2] = 1  # c01, c11
    
    # r=3: M4 = d(g-f)
    U[3][3] = 1               # d
    V[2][3] = 1; V[1][3] = -1 # g, -f
    W[0][3] = 1; W[1][3] = 1  # c00, c10
    
    # r=4: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[3][4] = 1               # h
    W[0][4] = -1; W[2][4] = 1 # c00, c01
    
    # r=5: M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1 # c, -a
    V[0][5] = 1; V[1][5] = 1  # e, f
    W[3][5] = 1               # c11
    
    # r=6: M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1 # b, -d
    V[2][6] = 1; V[3][6] = 1  # g, h
    W[0][6] = 1               # c00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
