import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0

    # The tensor is T[i,j,k] = 1 for all i,j,k (all-ones tensor)
    # Rank of all-ones tensor (n,m,p) is m (the middle dimension)
    # Decomposition: U[i,j,r] = delta_{j,r}, V[j,k,r] = delta_{j,r}, W[k,i,r] = 1
    rank = m  # = 3
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        # U[i, j, r] = 1 if j == r
        for i in range(n):
            U[i * m + r, r] = 1.0
        # V[j, k, r] = 1 if j == r
        for k in range(p):
            V[r * p + k, r] = 1.0
        # W[k, i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1.0
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if not np.isclose(val, 1.0, atol=1e-12):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: rank n*m*p
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback))
        V_f = np.zeros((m * p, rank_fallback))
        W_f = np.zeros((p * n, rank_fallback))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
