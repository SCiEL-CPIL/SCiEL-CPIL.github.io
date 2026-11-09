import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the standard schoolbook tensor decomposition for 5x5 matrix multiplication.
    Rank = 125.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros using NumPy for efficiency
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Iterate over all indices i, j, k to populate the decomposition
    # The column index corresponds to the triple (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                
                # U[i*m + j, col] = 1
                U[i * m + j, col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k, col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i, col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
