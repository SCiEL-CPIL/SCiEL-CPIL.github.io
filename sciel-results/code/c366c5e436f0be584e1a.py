import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook decomposition
    # Each term c corresponds to a unique (i, j, k) triple
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m + j) * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                
    # Convert to lists as per interface requirement if strict, 
    # but numpy arrays are usually accepted if "integer/half-integer entries" 
    # and shapes match. The prompt says "factor matrices of shapes...".
    # The original code returned lists of lists. I will return lists of lists to be safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
