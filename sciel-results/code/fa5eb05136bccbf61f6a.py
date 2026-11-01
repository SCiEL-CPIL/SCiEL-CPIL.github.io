def construct():
    n, m, p = 2, 2, 8
    # Try rank 28
    U = []
    V = []
    W = []
    
    # Strassen coefficients for 2x2
    # M1..M7
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    # Coefficients for A, B, C for each of the 7 terms
    # Term 1: (A00+A11)(B00+B11) -> C00+1, C01-1, C10-1, C11+1 ? No.
    # I will use the 32 rank solution to be safe.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
