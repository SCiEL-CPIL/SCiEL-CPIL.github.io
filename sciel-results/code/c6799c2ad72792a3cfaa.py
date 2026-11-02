def construct():
    n, m, p = 2, 2, 15
    rank = 7 * 7 + 4  # 7 blocks of Strassen (7 each) + 1 column of schoolbook (4)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 7 Strassen blocks for columns 0-13
    for t in range(7):
        c0 = 2 * t
        c1 = 2 * t + 1
        # M1 = A[0,0] * (B[0,c0] + B[1,c1])
        U[0][col] = 1
        V[0 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        # M2 = (A[0,0] + A[1,0]) * B[1,c1]
        U[0][col] = 1
        U[2][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        # M3 = (A[0,1] + A[1,1]) * B[0,c0]
        U[1][col] = 1
        U[3][col] = 1
        V[0 * p + c0][col] = 1
        W[c0 * n + 0][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        # M4 = A[1,1] * (B[0,c1] - B[1,c0])
        U[3][col] = 1
        V[0 * p + c1][col] = 1
        V[1 * p + c0][col] = -1
        W[c0 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        # M5 = (A[0,0] + A[0,1]) * B[1,c0]
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        # M6 = (A[1,0] - A[0,0]) * (B[0,c0] + B[0,c1])
        U[2][col] = 1
        U[0][col] = -1
        V[0 * p + c0][col] = 1
        V[0 * p + c1][col] = 1
        W[c0 * n + 0][col] = -1
        W[c0 * n + 1][col] = 1
        col += 1
        # M7 = (A[1,1] - A[1,0]) * (B[1,c0] + B[1,c1])
        U[3][col] = 1
        U[2][col] = -1
        V[1 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c1 * n + 0][col] = -1
        W[c1 * n + 1][col] = 1
        col += 1
    
    # Schoolbook for column 14
    c = 14
    # A[0,0] * B[0,c]
    U[0][col] = 1
    V[0 * p + c][col] = 1
    W[c * n + 0][col] = 1
    col += 1
    # A[0,1] * B[1,c]
    U[1][col] = 1
    V[1 * p + c][col] = 1
    W[c * n + 0][col] = 1
    col += 1
    # A[1,0] * B[0,c]
    U[2][col] = 1
    V[0 * p + c][col] = 1
    W[c * n + 1][col] = 1
    col += 1
    # A[1,1] * B[1,c]
    U[3][col] = 1
    V[1 * p + c][col] = 1
    W[c * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
