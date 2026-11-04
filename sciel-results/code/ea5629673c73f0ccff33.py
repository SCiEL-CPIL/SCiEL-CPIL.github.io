import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
                
    def to_list(X):
        return [[float(x) for x in row] for row in X]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
