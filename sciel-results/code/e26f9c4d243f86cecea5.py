"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with the trivial decomposition of rank n*m*p
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by merging columns
    # The tensor entries are all 1s in a structured way.
    # We can try to find linear dependencies among the columns.
    
    # For each pair of columns, check if one is a linear combination of the other
    # This is a simplification; a full rank reduction would require more work.
    
    # A better approach: use the known lower bounds and try to find a decomposition
    # with rank close to the lower bound.
    
    # Lower bound for matrix multiplication tensor rank:
    # R >= max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Also R >= (n*m*p)^(1/2) = sqrt(60) ≈ 7.75
    # And R >= max(n+m, m+p, p+n) - 1 = max(5, 13, 12) - 1 = 12
    
    # Let's try to find a decomposition with rank 30 (the obvious lower bound)
    # by using a structured approach.
    
    # For multiplying a 2x3 matrix by a 3x10 matrix:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can use the decomposition where each column corresponds to a (i,j,k) tuple,
    # but we can group them.
    
    # Actually, let's just try to verify if rank 30 is achievable.
    # The trivial decomposition has rank 60.
    # We need to find linear dependencies.
    
    # For now, return the trivial decomposition as a safe baseline.
    # A more sophisticated search would be needed to find a lower rank.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
