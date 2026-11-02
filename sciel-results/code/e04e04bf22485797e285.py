import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    
    # Use numpy for efficient initialization
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
