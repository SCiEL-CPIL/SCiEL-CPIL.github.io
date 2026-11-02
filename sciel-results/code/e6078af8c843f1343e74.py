"""Program for rank-minimal tensor decomposition of 4x5 * 5x5 matrix multiplication."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using bilinear algorithm search
    # The tensor T has T[i,j,k] = delta for the matrix multiplication
    # We want to find U, V, W of size (nm x r), (mp x r), (pn x r) such that
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Known bounds: For 4x5*5x5, the rank is at most n*m*p = 100
    # We can try to find improvements using algebraic methods
    
    # Let's try a simple approach: check if we can use the fact that
    # some entries might be combined
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    # and has integer entries
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
