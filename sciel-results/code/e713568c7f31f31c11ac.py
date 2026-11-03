DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    rank = n * p  # 36
    U = [[0] * rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * rank for _ in range(m * p)]  # 36 x 36
    W = [[0] * rank for _ in range(p * n)]  # 36 x 36
    
    for a in range(n):      # a represents i
        for b in range(p):  # b represents k
            c = a * p + b
            # U[i*m+j, c] = 1 iff i == a
            for j in range(m):
                U[a * m + j][c] = 1
            # V[j*p+k, c] = 1 iff k == b
            for j in range(m):
                V[j * p + b][c] = 1
            # W[k*n+i, c] = 1 iff i == a and k == b
            W[b * n + a][c] = 1
    
    return {"U": U, "V": V, "W": W}
