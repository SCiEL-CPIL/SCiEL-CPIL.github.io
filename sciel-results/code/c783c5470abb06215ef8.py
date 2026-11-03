import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists if required, or return numpy arrays?
    # The seed program returned lists. The prompt says "factor matrices".
    # Usually numpy arrays are accepted in these contexts.
    # However, to be strictly compatible with "integer/half-integer entries" and standard interfaces,
    # I'll return numpy arrays as they are standard in scientific Python.
    
    return {"U": U, "V": V, "W": W}
