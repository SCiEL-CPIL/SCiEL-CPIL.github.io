import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Schoolbook decomposition: rank = n * m * p = 4 * 4 * 9 = 144
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each (i, j, k):
    # C[i, j] += A[i, k] * B[k, j]
    # U index: i * m + k
    # V index: k * p + j
    # W index: j * n + i
    # Coefficient: 1
    
    idx = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                U[i * m + k, idx] = 1
                V[k * p + j, idx] = 1
                W[j * n + i, idx] = 1
                idx += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
