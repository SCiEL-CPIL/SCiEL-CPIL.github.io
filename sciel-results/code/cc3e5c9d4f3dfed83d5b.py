def construct():
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map index r to (i, j, k)
    # r = i * m * p + j * p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                # U index: i * m + j
                U[i * m + j][r] = 1
                # V index: j * p + k
                V[j * p + k][r] = 1
                # W index: k * n + i (as per problem description W[k*n+i,r])
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
