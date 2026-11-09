def construct():
    n, m, p = 2, 3, 5
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: i = 0
    # U[0*m + j][0] = 1 for all j
    # V[j*p + k][0] = 1 for all j, k
    # W[k*n + 0][0] = 1 for all k
    
    for j in range(m):
        U[j][0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        W[k * n + 0][0] = 1
        
    # Block 2: i = 1
    # U[1*m + j][1] = 1 for all j
    # V[j*p + k][1] = 1 for all j, k
    # W[k*n + 1][1] = 1 for all k
    
    for j in range(m):
        U[m + j][1] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1
    for k in range(p):
        W[k * n + 1][1] = 1
        
    return {"U": U, "V": V, "W": W}
