import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-27 (schoolbook) decomposition for 3x3 matrix multiplication.
    Uses NumPy arrays for efficiency and compatibility.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indexing based on convention:
                # U[i*m + j, column]
                # V[j*p + k, column]
                # W[k*n + i, column]
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
