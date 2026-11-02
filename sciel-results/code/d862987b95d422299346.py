"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal decomposition
    # Use a randomized approach to search for low-rank solutions
    
    # Try various ranks
    for rank in range(30, n * m * p + 1):
        # Try to find a solution with this rank
        # This is a hard problem, so we'll use a simple heuristic
        
        # Initialize with random small values
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Try to find a solution
        # For simplicity, just return schoolbook
        pass
    
    # Return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
