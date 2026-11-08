DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    rank = p * m  # 9 * 4 = 36
    U = [[0] * rank for _ in range(n * m)]  # (16, 36)
    V = [[0] * rank for _ in range(m * p)]  # (36, 36)
    W = [[0] * rank for _ in range(p * n)]  # (36, 36)
    
    for k in range(p):  # k = 0..8
        for j in range(m):  # j = 0..3
            c = k * m + j  # column index
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
