import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank is m = 2
    # U: (n*m, rank) = (4, 2)
    # V: (m*p, rank) = (28, 2)
    # W: (p*n, rank) = (28, 2)
    
    rank = m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i, j, r] = 1 if j == r
    # Index in U is i*m + j
    for i in range(n):
        for j in range(m):
            if j < rank: # rank is m, so j is always < rank
                U[i * m + j, j] = 1
                
    # V[j, k, r] = 1 if j == r
    # Index in V is j*p + k
    for j in range(m):
        for k in range(p):
            if j < rank:
                V[j * p + k, j] = 1
                
    # W[k, i, r] = 1 for all k, i, r
    # Index in W is k*n + i
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
