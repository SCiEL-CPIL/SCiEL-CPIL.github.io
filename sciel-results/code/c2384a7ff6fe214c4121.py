# EVOLVE-BLOCK-START
import time as _time

def construct():
    n, m, p = 3, 3, 6
    # Rank-m decomposition: for each j in 0..m-1, one term with
    # U[i*m+j, j] = 1, V[j*p+k, j] = 1, W[k*n+i, j] = 1
    # This gives sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
