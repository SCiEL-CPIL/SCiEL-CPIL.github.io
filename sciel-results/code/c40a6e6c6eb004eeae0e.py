import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 10
    
    # Schoolbook decomposition for M_{2,3,10}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U index: i*m + j (i in 0..1, j in 0..2)
    # V index: j*p + k (j in 0..2, k in 0..9)
    # W index: k*n + i (k in 0..9, i in 0..1) per trace convention W[k*n+i, r]
    
    rank = n * m * p  # 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
