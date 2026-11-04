def construct():
    n, m, p = 2, 4, 7
    rank = 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U[i*m + j, r] = 1 if j == r else 0
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if j == r:
                    U[i*m + j][r] = 1
                    
    # V[j*p + k, r] = 1 if j == r else 0
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if j == r:
                    V[j*p + k][r] = 1
                    
    # W[k*n + i, r] = 1
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k*n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
