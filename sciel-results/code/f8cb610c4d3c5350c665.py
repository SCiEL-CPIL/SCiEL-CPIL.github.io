import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for matrix multiplication 2x3 by 3x6.
    Returns a dictionary with keys 'U', 'V', 'W' containing NumPy arrays.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Fill in the schoolbook decomposition
    # Each column c corresponds to a unique (i, j, k) triplet
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by row i and col j of A (flattened i*m+j)
                U[i * m + j, column] = 1
                # V is indexed by row j and col k of B (flattened j*p+k)
                V[j * p + k, column] = 1
                # W is indexed by row k and col i of C (flattened k*n+i)
                # Note: Task specifies W[k*n+i, r]
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
