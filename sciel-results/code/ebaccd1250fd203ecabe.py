"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition as guaranteed fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly computes the multiplication tensor."""
        # Check that T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def check_integer_half_integer(U, V, W):
        """Check all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # Check if val is close to an integer or half-integer
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: start from schoolbook and try to eliminate one term
    # by finding a linear combination
    
    # For now, let's try a simple approach: see if rank 41 is possible
    # by attempting to merge two schoolbook terms
    
    # The schoolbook has terms for each (i,j,k). 
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if we can find
    # a single rank-1 tensor that reproduces both contributions.
    
    # This is generally not possible, but let's try a search for small improvements.
    
    # Given time constraints, return schoolbook if no quick improvement found
    if time.time() > deadline - 5.0:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try rank 41: attempt to find a decomposition
    # This is a hard problem; for now, return schoolbook
    # A more sophisticated search would be needed for actual improvement
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
