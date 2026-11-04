import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 72: 8 blocks of Strassen (7 each) + 16 scalar products for last col
    rank = 56 + 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2
    # u, v, w are 4-vectors (local indices 0,1,2,3)
    # u: X11, X12, X21, X22
    # v: Y11, Y12, Y21, Y22
    # w: C11, C12, C21, C22
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 0, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 1, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # 8 blocks
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for s in range(7):
                    # Map local indices to global
                    # u: i (0-1), j (0-1) -> global i, j
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            val = strassen_u[s][local_idx]
                            if val != 0:
                                gi = I * 2 + li
                                gj = J * 2 + lj
                                U[gi * m + gj, col] += val
                                
                    # v: j (0-1), k (0-1) -> global j, k
                    for lj in range(2):
                        for lk in range(2):
                            local_idx = lj * 2 + lk
                            val = strassen_v[s][local_idx]
                            if val != 0:
                                gj = J * 2 + lj
                                gk = K * 2 + lk
                                V[gj * p + gk, col] += val
                                
                    # w: k (0-1), i (0-1) -> global k, i
                    # Note: W index is k*n + i
                    for lk in range(2):
                        for li in range(2):
                            local_idx = lk * 2 + li
                            val = strassen_w[s][local_idx]
                            if val != 0:
                                gk = K * 2 + lk
                                gi = I * 2 + li
                                W[gk * n + gi, col] += val
                                
                    col += 1
                    
    # 16 scalar products for k=4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
