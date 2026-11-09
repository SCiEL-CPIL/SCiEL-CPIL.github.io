import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for 5x5 matrix multiplication.
    Strategy: Direct sum of 25 independent rank-5 components (Schoolbook).
    Rank = 125.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists to match seed interface
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct decomposition
    # Each term corresponds to a specific (i, j, k) interaction
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s in the factor matrices
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
