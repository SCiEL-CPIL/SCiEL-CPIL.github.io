import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = n * m * p
    
    # Create identity-like tensors
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k)
    # We can use broadcasting
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Vectorized construction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                
    return {"U": U, "V": V, "W": W}
