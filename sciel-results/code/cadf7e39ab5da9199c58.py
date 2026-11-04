import numpy as np

DIMENSIONS = [4, 5, 5]

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def combine(U1, V1, W1, U2, V2, W2):
    # Concatenate along the rank dimension (columns)
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Direct schoolbook decomposition for the full problem
    U, V, W = schoolbook(n, m, p)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
