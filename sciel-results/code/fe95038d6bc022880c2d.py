def construct():
    n, m, p = 2, 2, 12
    rank = 42  # 7 * 6 blocks
    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)
    
    for t in range(6):
        base = t * 7
        # Columns 2t and 2t+1
        c0 = 2*t
        c1 = 2*t+1
        
        # M1 = A11*(B12-B22)
        col = base
        U[0][col] = 1  # A11
        V[0*12+c1][col] = 1  # B12 = B[0, c1]
        V[1*12+c1][col] = -1  # B22 = B[1, c1]
        W[c0*2+0][col] = 1  # C11 += M1
        W[c1*2+0][col] = 1  # C12 += M1
        
        # M2 = (A11+A12)*B22
        col = base+1
        U[0][col] = 1  # A11
        U[1][col] = 1  # A12
        V[1*12+c1][col] = 1  # B22
        W[c1*2+1][col] = 1  # C22 += M2... wait
        
        # I need to be more careful with the W indexing.
        # C[i,k] has W-index k*n+i = k*2+i
        # C11 = C[0,0] → W-index 0*2+0 = 0
        # C12 = C[0,1] → W-index 1*2+0 = 2
        # C21 = C[1,0] → W-index 0*2+1 = 1
        # C22 = C[1,1] → W-index 1*2+1 = 3
        
        # So for block t, the W-indices are:
        # C[0,2t] → (2t)*2+0 = 4t
        # C[0,2t+1] → (2t+1)*2+0 = 4t+2
        # C[1,2t] → (2t)*2+1 = 4t+1
        # C[1,2t+1] → (2t+1)*2+1 = 4t+3
        
        pass
    
    return {"U": U, "V": V, "W": W}
