def _rank4_decomposition():
    n, m, p = 4, 4, 9
    r = 4
    U = [[0.0]*r for _ in range(n*m)]  # 16 x 4
    V = [[0.0]*r for _ in range(m*p)]  # 36 x 4
    W = [[0.0]*r for _ in range(p*n)]  # 36 x 4
    
    for i in range(n):
        for j in range(m):
            U[i*m+j][i] = 1.0  # U[i*m+j, c] = delta_{i,c}
    
    for j in range(m):
        for k in range(p):
            for c in range(r):
                V[j*p+k][c] = 1.0  # V is all ones
    
    for k in range(p):
        for i in range(n):
            if k < r:
                W[k*n+i][k] = 1.0  # W[k*n+i, c] = delta_{k,c} for k<4
    
    return {"U": U, "V": V, "W": W}
