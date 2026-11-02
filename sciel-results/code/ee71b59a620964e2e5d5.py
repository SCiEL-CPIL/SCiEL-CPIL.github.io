"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try to find a low-rank decomposition
    # Known: schoolbook rank = n*m*p = 96
    # Try to use known results or search
    
    # For 3x4x8, let's try a construction based on splitting
    # We can write C = A*B where A is 3x4, B is 4x8
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try rank = 40 (guess based on known bounds)
    # Use a heuristic: try to find a decomposition with rank < 96
    
    # Approach: use the fact that we can combine terms
    # For now, return schoolbook if search fails
    
    # Let's try a specific known construction
    # For 3x4x8, one approach is to use the "lattice" method
    
    # Actually, let's try to verify if rank 48 works by construction
    # Split the 8 columns into two groups of 4
    # C[:, 0:4] = A @ B[:, 0:4] and C[:, 4:8] = A @ B[:, 4:8]
    # Each 3x4x4 has schoolbook rank 48, so total would be 96 - no help
    
    # Try a different approach: use Strassen-like ideas
    # For 2x2x2, rank 7 saves 1 term
    # For 3x3x3, rank 23 saves 4 terms
    
    # Let's try to use a computational search
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want to find the minimum r such that T can be written as
    # sum_{c=0}^{r-1} U[:,c] outer V[:,c] outer W[:,c]
    
    # This is a hard problem. Let's try a randomized local search.
    
    # For the time being, return schoolbook
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
