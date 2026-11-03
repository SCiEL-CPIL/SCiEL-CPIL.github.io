import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 4
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        for i in range(n):
            U[i * m + r, r] = 1
        for l in range(p):
            V[r * p + l, r] = 1
        for l in range(p):
            for i in range(n):
                W[l * n + i, r] = 1
                
    return {"U": U, "V": V, "W": W}
