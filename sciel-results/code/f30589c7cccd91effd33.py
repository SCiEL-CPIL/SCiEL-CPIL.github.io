import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = n * m * p  # 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    
    return {'U': U, 'V': V, 'W': W}
