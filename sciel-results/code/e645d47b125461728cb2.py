def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = 2 * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 24
    V = [[0] * rank for _ in range(m * p)]   # 24 x 24
    W = [[0] * rank for _ in range(p * n)]   # 24 x 24
    
    for k in range(p):
        # Term 2k: row 0 of C
        col = 2 * k
        U[0 * m + 0][col] = 1  # A[0,0]
        U[0 * m + 1][col] = 1  # A[0,1]
        V[0 * p + k][col] = 1  # B[0,k]
        V[1 * p + k][col] = 1  # B[1,k]
        W[k * n + 0][col] = 1  # C[0,k]
        
        # Term 2k+1: row 1 of C
        col = 2 * k + 1
        U[1 * m + 0][col] = 1  # A[1,0]
        U[1 * m + 1][col] = 1  # A[1,1]
        V[0 * p + k][col] = 1  # B[0,k]
        V[1 * p + k][col] = 1  # B[1,k]
        W[k * n + 1][col] = 1  # C[1,k]
    
    return {"U": U, "V": V, "W": W}
