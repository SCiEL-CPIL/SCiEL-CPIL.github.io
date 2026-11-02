import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    a_dim = n * m  # 8
    b_dim = m * p  # 20
    c_dim = p * n  # 10
    
    rank = m * p  # 20
    
    U = np.zeros((a_dim, rank))
    V = np.zeros((b_dim, rank))
    W = np.zeros((c_dim, rank))
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
