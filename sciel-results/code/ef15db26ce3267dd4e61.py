import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to use a better decomposition for 2x2xp blocks
    # Rank of 2x2xp is suspected to be 3p+1 for small p
    # For p=6, rank = 19
    # Two blocks -> 38
    
    # We will implement the schoolbook for now to ensure correctness
    # and structure the code to allow swapping in a better decomposition
    
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
