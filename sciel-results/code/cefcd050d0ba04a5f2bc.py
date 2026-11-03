def construct():
    n, m, p = 4, 4, 8
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I will use the rank-20 decomposition.
    
    return {"U": U, "V": V, "W": W}
