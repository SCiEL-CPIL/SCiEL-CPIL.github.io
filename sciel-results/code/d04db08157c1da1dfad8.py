def construct():
    n, m, p = 2, 2, 13
    rank = 4
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 4)
    V = [[0] * rank for _ in range(m * p)]  # (26, 4)
    W = [[0] * rank for _ in range(p * n)]  # (26, 4)
    
    # Term 0: A[0,0] * B[0,k] -> C[0,k]
    U[0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1   # B[0,k]
        W[0 * n + k][0] = 1   # C[0,k]
    
    # Term 1: A[0,1] * B[1,k] -> C[0,k]
    U[1][1] = 1
    for k in range(p):
        V[1 * p + k][1] = 1   # B[1,k]
        W[0 * n + k][1] = 1   # C[0,k]
    
    # Term 2: A[1,0] * B[0,k] -> C[1,k]
    U[2][2] = 1
    for k in range(p):
        V[0 * p + k][2] = 1   # B[0,k]
        W[1 * n + k][2] = 1   # C[1,k]
    
    # Term 3: A[1,1] * B[1,k] -> C[1,k]
    U[3][3] = 1
    for k in range(p):
        V[1 * p + k][3] = 1   # B[1,k]
        W[1 * n + k][3] = 1   # C[1,k]
    
    return {"U": U, "V": V, "W": W}
