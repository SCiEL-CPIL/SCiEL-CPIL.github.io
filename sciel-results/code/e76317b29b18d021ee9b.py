def construct():
    n, m, p = 2, 2, 11
    rank = 2 * p  # 22
    U = [[0] * rank for _ in range(n * m)]  # (4, 22)
    V = [[0] * rank for _ in range(m * p)]  # (22, 22)
    W = [[0] * rank for _ in range(p * n)]  # (22, 22)
    
    for k in range(p):
        # Term 2k: handles B[0,k]
        for i in range(n):
            U[i * m + 0][2 * k] = 1
            W[k * n + i][2 * k] = 1
        V[0 * p + k][2 * k] = 1
        
        # Term 2k+1: handles B[1,k]
        for i in range(n):
            U[i * m + 1][2 * k + 1] = 1
            W[k * n + i][2 * k + 1] = 1
        V[1 * p + k][2 * k + 1] = 1
    
    return {"U": U, "V": V, "W": W}
