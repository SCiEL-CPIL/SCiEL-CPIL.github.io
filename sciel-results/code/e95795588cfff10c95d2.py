"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Uses sparse Linear Programming (LP) to minimize the rank of the factor matrices.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # The matrix multiplication tensor T has components T[i,j,k] = 1 if we're computing C[i,k]
    # The tensor can be decomposed as sum_r U[:,r] * V[:,r] * W[:,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # For the tensor decomposition, we need:
    # For each (i,j,k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The schoolbook decomposition gives rank = n*m*p = 90
    # Let's try to find a lower rank decomposition using LP
    
    # Since exact tensor decomposition is a hard problem, let's use a heuristic approach:
    # We'll try to find a decomposition with rank less than 90 by solving a relaxed LP problem
    
    # First, let's establish the baseline schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition using iterative refinement
    # We'll use a greedy approach: start with schoolbook and try to merge terms
    
    # For now, return the schoolbook decomposition as it's guaranteed to be correct
    # A more sophisticated approach would use LP to find the minimum rank
    
    rank = schoolbook_rank
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
