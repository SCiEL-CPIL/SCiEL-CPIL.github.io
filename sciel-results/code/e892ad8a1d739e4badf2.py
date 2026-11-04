import numpy as np

def construct():
    n, m, p = 4, 5, 6
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for bi in range(2):  # row block
        for bj in range(3):  # col block
            for k in range(m):  # middle index
                for di in range(2):  # within row block
                    i = 2*bi + di
                    U[i*m + k, col] = 1
                for dj in range(2):  # within col block
                    j = 2*bj + dj
                    V[k*p + j, col] = 1
                    for di in range(2):
                        i = 2*bi + di
                        W[j*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
