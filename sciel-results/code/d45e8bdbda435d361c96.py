"""Matrix multiplication 2x3 by 3x4 tensor decomposition."""
DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 12 x 24
    W = [[0] * rank for _ in range(p * n)]  # 8 x 24
    
    # Schoolbook: for each (i,j,k), set U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
