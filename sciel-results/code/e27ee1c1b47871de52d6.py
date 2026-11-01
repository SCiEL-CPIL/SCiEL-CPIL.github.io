import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Direct decomposition: rank = n * m * p = 4 * 13 = 52
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A flat index: i*m + j = 2*i + j
    # B flat index: j*p + k = 13*j + k
    # C flat index (W): k*n + i = 2*k + i
    
    rank = n * m * p  # 52
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
