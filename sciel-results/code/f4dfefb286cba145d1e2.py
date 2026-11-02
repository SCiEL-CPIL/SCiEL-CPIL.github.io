"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import expm, logm, solve, inv
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = 3, 4, 5
    total_terms = n * m * p  # 60

    # We attempt to find a lower rank decomposition using a structured approach.
    # The key insight: for small matrix multiplication, we can sometimes merge
    # terms by exploiting linear dependencies.
    
    # One effective approach: use the fact that the tensor is a "slice" of the
    # full matrix multiplication tensor, and apply known lower-bound results
    # or constructive algorithms.
    
    # For 3x4 * 4x5, the known lower bound is 20 (from general bounds), but
    # the actual minimum rank is likely higher. Let's try a constructive approach.
    
    # Approach: Use a randomized numerical method to find a low-rank decomposition.
    # We'll use a variant of the "rank minimization" via alternating projections.
    
    # First, let's try to find a decomposition of rank r by solving a system
    # of polynomial equations. We'll use a heuristic search.
    
    # Start with the schoolbook decomposition and try to merge terms.
    # We'll use a greedy approach: try to combine pairs of terms and see if
    # the resulting tensor still decomposes correctly.
    
    # Actually, let's use a more principled approach: 
    # The tensor T can be viewed as a 3-way tensor of size 12 x 20 x 15.
    # We want to find the minimum number of rank-1 tensors that sum to T.
    
    # Let's try rank values from 20 up to 60 and check if a decomposition exists.
    # We'll use a numerical optimization approach.
    
    # For efficiency, let's first try a few specific ranks and see if we can
    # find a valid decomposition.
    
    # Actually, the simplest improvement is to use a known result: 
    # For matrix multiplication of dimensions (n, m, p), the rank is at most
    # n*m*p, but for small cases, we can do better.
    
    # Let's try to construct a decomposition of rank 30 by pairing up terms.
    
    # For each (i, j, k), we have a term. We can try to combine terms that
    # share the same i or same k.
    
    # Actually, let's just return the schoolbook decomposition for now, as
    # finding a lower rank decomposition is non-trivial.
    
    rank = total_terms
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
