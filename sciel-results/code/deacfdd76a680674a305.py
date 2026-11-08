"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    Dimensions: (2, 3, 5)
    Rank: 30
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the corresponding entries in U, V, W to 1
                U[i * m + j, col_idx] = 1.0
                V[j * p + k, col_idx] = 1.0
                W[k * n + i, col_idx] = 1.0
                col_idx += 1
                
    # Convert to lists to ensure compatibility with the expected interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
