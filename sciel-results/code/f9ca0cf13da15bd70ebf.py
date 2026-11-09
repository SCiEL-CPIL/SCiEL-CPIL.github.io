import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
    
    # Try to reduce rank by merging terms
    # This is a placeholder for a more sophisticated search
    # For now, return the schoolbook decomposition
    # Convert to lists for the required format
    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
