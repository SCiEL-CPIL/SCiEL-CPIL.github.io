import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split the p=6 columns into two groups of 3.
    # Each group is a 3x3 multiplication.
    # Rank for each 3x3 is 9 (schoolbook).
    # Total rank = 18.
    
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # First block: k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Second block: k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                k_actual = k + 3
                U[i * m + j, col] = 1
                V[j * p + k_actual, col] = 1
                W[k_actual * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
