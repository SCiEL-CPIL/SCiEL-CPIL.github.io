def construct():
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (per prompt: W[k*n+i, r])
                
                U[i*m + j][r] = 1
                V[j*p + k][r] = 1
                W[k*n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
