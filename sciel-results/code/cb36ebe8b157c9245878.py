def construct():
    n, m, p = 2, 2, 13
    
    # Column pairs for Strassen: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    # Single column for schoolbook: 12
    
    rank = 6 * 7 + 4  # 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    col = 0
    
    # Strassen blocks for column pairs
    for k0 in range(0, 12, 2):
        k1 = k0 + 1
        
        # m1 = (A00+A11) * (B0k0+B1k1)
        U[0][col] += 1   # A[0,0]
        U[3][col] += 1   # A[1,1]
        V[0*p+k0][col] += 1   # B[0,k0]
        V[1*p+k1][col] += 1   # B[1,k1]
        W[k0*n+0][col] += 1   # C[k0,0]
        W[k1*n+1][col] += 1   # C[k1,1]
        col += 1
        
        # m2 = (A10+A11) * B0k0
        U[2][col] += 1   # A[1,0]
        U[3][col] += 1   # A[1,1]
        V[0*p+k0][col] += 1   # B[0,k0]
        W[k0*n+1][col] += 1   # C[k0,1]
        W[k1*n+1][col] += -1  # C[k1,1]
        col += 1
        
        # m3 = A00 * (B0k1-B1k0)
        U[0][col] += 1   # A[0,0]
        V[0*p+k1][col] += 1   # B[0,k1]
        V[1*p+k0][col] += -1  # B[1,k0]
        W[k1*n+0][col] += 1   # C[k1,0]
        W[k1*n+1][col] += 1   # C[k1,1]
        col += 1
        
        # m4 = A11 * (B1k0-B0k0)
        U[3][col] += 1   # A[1,1]
        V[1*p+k0][col] += 1   # B[1,k0]
        V[0*p+k0][col] += -1  # B[0,k0]
        W[k0*n+0][col] += 1   # C[k0,0]
        W[k0*n+1][col] += 1   # C[k0,1]
        col += 1
        
        # m5 = (A00+A01) * B1k1
        U[0][col] += 1   # A[0,0]
        U[1][col] += 1   # A[0,1]
        V[1*p+k1][col] += 1   # B[1,k1]
        W[k0*n+0][col] += -1  # C[k0,0]
        W[k1*n+0][col] += 1   # C[k1,0]
        col += 1
        
        # m6 = (A10-A00) * (B0k0+B0k1)
        U[2][col] += 1   # A[1,0]
        U[0][col] += -1  # A[0,0]
        V[0*p+k0][col] += 1   # B[0,k0]
        V[0*p+k1][col] += 1   # B[0,k1]
        W[k1*n+1][col] += 1   # C[k1,1]
        col += 1
        
        # m7 = (A01-A11) * (B1k0+B1k1)
        U[1][col] += 1   # A[0,1]
        U[3][col] += -1  # A[1,1]
        V[1*p+k0][col] += 1   # B[1,k0]
        V[1*p+k1][col] += 1   # B[1,k1]
        W[k0*n+0][col] += 1   # C[k0,0]
        col += 1
    
    # Schoolbook column for k=12
    k = 12
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] += 1
            V[j*p+k][col] += 1
            W[k*n+i][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
