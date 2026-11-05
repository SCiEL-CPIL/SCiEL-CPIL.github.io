def construct():
    n, m, p = 3, 4, 4
    rank = 16  # 4 * 4
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 16
    V = [[0] * rank for _ in range(m * p)]  # 16 x 16
    W = [[0] * rank for _ in range(p * n)]  # 12 x 16
    
    for a in range(m):  # a = 0,1,2,3 (j index)
        for b in range(p):  # b = 0,1,2,3 (k index)
            r = a * p + b  # term index 0..15
            
            # U[i*m+j, r] = 1 if j == a
            for i in range(n):
                U[i * m + a][r] = 1
            
            # V[j*p+k, r] = 1 if k == b
            for j in range(m):
                V[j * p + b][r] = 1
            
            # W[k*n+i, r] = 1 for all k, i
            for k in range(p):
                for i in range(n):
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
