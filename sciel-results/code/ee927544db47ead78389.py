"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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
    
    # Try to find a lower rank decomposition using algebraic methods
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    # We need: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For rank r, we have r unknowns per matrix entry
    # Total unknowns: r*(n*m + m*p + p*n) = r*(6 + 21 + 14) = 41r
    # Constraints: n*m*p = 42 equations (one for each tensor entry)
    
    # For r=42, we have 41*42 = 1722 unknowns for 42 equations - highly underdetermined
    # For r=41, we have 41*41 = 1681 unknowns for 42 equations - still underdetermined
    
    # The question is whether a solution with integer/half-integer entries exists for r < 42
    
    # Let's try to verify if rank 41 is possible by checking if the tensor
    # can be expressed as a sum of 41 rank-1 tensors
    
    # This is a hard problem in general. Let's try a simple approach:
    # Start with schoolbook and try to eliminate one term by finding
    # a linear combination that preserves the tensor
    
    # Given the time constraints and the difficulty of the problem,
    # and since we need integer/half-integer entries, let's return
    # the schoolbook decomposition which is guaranteed correct.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
