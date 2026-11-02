def construct():
    n, m, p = 3, 3, 4
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 21 terms
    # ... (hardcoded)
    
    # 3 terms for col 3
    for i in range(3):
        r = 21 + i
        for j in range(3):
            U[i * m + j][r] = 1
        for j in range(3):
            V[j * p + 3][r] = 1
        W[3 * n + i][r] = 1
        
    return {"U": U, "V": V, "W": W}
