"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 7 + 4 = 53
    rank = 53
    
    # U shape: (n*m, rank) = (4, 53)
    # V shape: (m*p, rank) = (30, 53)
    # W shape: (p*n, rank) = (30, 53)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 7 Strassen blocks for p=14 (k=0 to 13)
    for b in range(7):
        k = 2 * b
        
        # M1 = (a+d)(e+h)
        # a=A[0,0]=U[0], d=A[1,1]=U[3]
        # e=B[0,k]=V[k], h=B[1,k+1]=V[15+k+1]
        # C[0,k]=W[2k], C[1,k+1]=W[2k+3]
        U[0][col] = 1
        U[3][col] = 1
        V[k][col] = 1
        V[15 + k + 1][col] = 1
        W[2 * k][col] = 1
        W[2 * k + 3][col] = 1
        col += 1
        
        # M2 = (c+d)e
        # c=A[1,0]=U[2], d=A[1,1]=U[3]
        # e=B[0,k]=V[k]
        # C[1,k]=W[2k+1], C[1,k+1]=W[2k+3]
        U[2][col] = 1
        U[3][col] = 1
        V[k][col] = 1
        W[2 * k + 1][col] = 1
        W[2 * k + 3][col] = 1
        col += 1
        
        # M3 = a(f-h)
        # a=A[0,0]=U[0]
        # f=B[0,k+1]=V[k+1], h=B[1,k+1]=V[15+k+1]
        # C[0,k+1]=W[2k+2], C[1,k+1]=W[2k+3]
        U[0][col] = 1
        V[k + 1][col] = 1
        V[15 + k + 1][col] = -1
        W[2 * k + 2][col] = 1
        W[2 * k + 3][col] = -1
        col += 1
        
        # M4 = d(g-e)
        # d=A[1,1]=U[3]
        # g=B[1,k]=V[15+k], e=B[0,k]=V[k]
        # C[0,k]=W[2k], C[1,k]=W[2k+1]
        U[3][col] = 1
        V[15 + k][col] = 1
        V[k][col] = -1
        W[2 * k][col] = -1
        W[2 * k + 1][col] = 1
        col += 1
        
        # M5 = (a+b)h
        # a=A[0,0]=U[0], b=A[0,1]=U[1]
        # h=B[1,k+1]=V[15+k+1]
        # C[0,k+1]=W[2k+2], C[1,k+1]=W[2k+3]
        U[0][col] = 1
        U[1][col] = 1
        V[15 + k + 1][col] = 1
        W[2 * k + 2][col] = 1
        W[2 * k + 3][col] = -1
        col += 1
        
        # M6 = (c-a)(e+f)
        # c=A[1,0]=U[2], a=A[0,0]=U[0]
        # e=B[0,k]=V[k], f=B[0,k+1]=V[k+1]
        # C[0,k]=W[2k]
        U[0][col] = -1
        U[2][col] = 1
        V[k][col] = 1
        V[k + 1][col] = 1
        W[2 * k][col] = -1
        col += 1
        
        # M7 = (b-d)(g+h)
        # b=A[0,1]=U[1], d=A[1,1]=U[3]
        # g=B[1,k]=V[15+k], h=B[1,k+1]=V[15+k+1]
        # C[0,k]=W[2k]
        U[1][col] = 1
        U[3][col] = -1
        V[15 + k][col] = 1
        V[15 + k + 1][col] = 1
        W[2 * k][col] = 1
        col += 1

    # 1 Standard block for p=1 (k=14)
    k = 14
    
    # M1 = A[0,0] * B[0,14]
    # U[0], V[14], W[28]
    U[0][col] = 1
    V[14][col] = 1
    W[28][col] = 1
    col += 1
    
    # M2 = A[0,1] * B[1,14]
    # U[1], V[29], W[28]
    U[1][col] = 1
    V[29][col] = 1
    W[28][col] = 1
    col += 1
    
    # M3 = A[1,0] * B[0,14]
    # U[2], V[14], W[29]
    U[2][col] = 1
    V[14][col] = 1
    W[29][col] = 1
    col += 1
    
    # M4 = A[1,1] * B[1,14]
    # U[3], V[29], W[29]
    U[3][col] = 1
    V[29][col] = 1
    W[29][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
