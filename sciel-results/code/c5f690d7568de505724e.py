def construct():
    n, m, p = 2, 3, 6
    rank = 22
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Rank 11 decomposition for 2x3x3
    # ...
    # Fill U, V, W
    return {"U": U, "V": V, "W": W}
