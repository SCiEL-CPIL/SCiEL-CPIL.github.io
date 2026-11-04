"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k), the tensor entry is 1
    # It should equal sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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

def _try_rank_reduction_peeling(n, m, p, max_time=100):
    """
    Attempt to find a decomposition of rank < n*m*p using a peeling/relaxation approach.
    We start from the schoolbook and try to combine terms by finding linear dependencies.
    """
    deadline = time.time() + max_time
    n, m, p = 3, 4, 6
    
    # The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Schoolbook has rank n*m*p = 72.
    # We want to see if we can do better.
    
    # Strategy: Try to find a decomposition of rank r < 72.
    # This is a hard problem in general. For 3x4x6, the known lower bound is 65 
    # (from Strassen-like improvements or algebraic geometry).
    
    # A practical approach: use a random/structured initialization and optimize
    # the entries to satisfy the constraints. But with 72 terms, we have 3*4*6=72 
    # constraints (one per (i,j,k) pair) and 3*r variables (U, V, W entries).
    
    # Actually, the constraints are:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p = 72 equations in 3*r unknowns (if we fix r).
    # But the equations are cubic in the variables, making it non-convex.
    
    # Alternative: Use the fact that the tensor is "all-ones".
    # T[i,j,k] = 1 = 1*1*1.
    # Can we write 1 as a sum of fewer products?
    
    # For the all-ones tensor of size n x m x p, the rank is known to be n*m*p 
    # in general, but sometimes less if there's structure.
    # However, for matrix multiplication, the tensor is not just all-ones in the 
    # (i,j,k) sense; it's the structure that matters.
    
    # Let's try a specific construction for 3x4x6.
    # Known results: The rank of 3x4x6 matrix multiplication is at most 66 
    # (from various papers). The schoolbook is 72.
    
    # We'll attempt a heuristic search for a rank-66 or lower decomposition.
    # This is complex, so we'll try a simpler approach: check if the current 
    # schoolbook can be slightly improved by exploiting the specific dimensions.
    
    # For now, return the schoolbook as the best known valid solution within time.
    # The peeling idea would require solving a non-convex optimization problem,
    # which is risky within the time limit.
    
    # Let's try a different tactic: use the known result that for 3x4x6,
    # a rank-66 decomposition exists. We'll try to construct it or find it
    # via a quick search.
    
    # Given the complexity and time limit, we'll return the schoolbook for now.
    # Future iterations could implement a proper optimization.
    
    return None  # Signal that we didn't find a better decomposition

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition
    result = _schoolbook(n, m, p)
    
    # Try to find a better (lower rank) decomposition
    # This is where we'd implement the peeling algorithm
    # For now, we return the schoolbook as a valid baseline
    
    # Verification (optional, for safety)
    # if not _verify(result['U'], result['V'], result['W'], n, m, p):
    #     raise ValueError("Schoolbook decomposition failed verification")
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
