def construct():
    n, m, p = 2, 4, 4
    rank = 2
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Rank 1 term for i=0
    for i in range(n):
        if i == 0:
            for j in range(m):
                U[i*m+j][0] = 1.0
        # Rank 2 term for i=1
        if i == 1:
            for j in range(m):
                U[i*m+j][1] = 1.0
                
    # V is all ones for both terms
    for j in range(m):
        for k in range(p):
            V[j*p+k][0] = 1.0
            V[j*p+k][1] = 1.0
            
    # W is all ones for both terms
    for k in range(p):
        for i in range(n):
            W[k*n+i][0] = 1.0
            W[k*n+i][1] = 1.0
            
    return {"U": U, "V": V, "W": W}
