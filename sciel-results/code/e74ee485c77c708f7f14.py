"""Matrix multiplication 4x4x5 decomposition."""
DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    # Schoolbook decomposition: rank = n*m*p = 80
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    rank = n * m * p  # 80
    U = [[0] * rank for _ in range(n * m)]  # 16 x 80
    V = [[0] * rank for _ in range(m * p)]  # 20 x 80
    W = [[0] * rank for _ in range(p * n)]  # 20 x 80
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
