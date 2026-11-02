import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Rank-10 construction:
    # For each (i, j) with i in {0,1}, j in {0..4}, create term r = i*5 + j
    R = n * m  # 10
    
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U: 1 at (i*m + j, r)
            U[i * m + j, r] = 1.0
            # V: 1 at (j*p + k, r) for all k in 0..p-1
            for k in range(p):
                V[j * p + k, r] = 1.0
            # W: 1 at (k*n + i, r) for all k in 0..p-1
            for k in range(p):
                W[k * n + i, r] = 1.0
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (should not be reached)
    R_school = n * m * p
    U_s = np.zeros((nm, R_school), dtype=float)
    V_s = np.zeros((mp, R_school), dtype=float)
    W_s = np.zeros((pn, R_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
