"""
Rewritten program for matrix multiplication tensor decomposition.
Task: 3x4 by 4x6 matrix multiplication.
Dimensions: n=3, m=4, p=6.
Schoolbook rank: 72.
"""
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construct schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists of lists to match interface requirement if strictly needed, 
    # but numpy arrays are often accepted. The prompt says "factor matrices of shapes...".
    # The seed used lists. I will return lists to be safe with the interface definition.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
