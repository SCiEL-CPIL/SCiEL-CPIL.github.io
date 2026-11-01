"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    
    We construct a valid decomposition using the trivial basis expansion.
    Each term in the matrix multiplication sum corresponds to a unique rank component.
    This results in a rank of n*m*p = 24.
    """
    n, m, p = DIMENSIONS
    
    # Total number of terms in the multiplication
    num_terms = n * m * p
    rank = num_terms
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices
    # r corresponds to the term (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                # U index: i*m + j
                U[i * m + j][r] = 1
                
                # V index: j*p + k
                V[j * p + k][r] = 1
                
                # W index: k*n + i (as per task description)
                W[k * n + i][r] = 1
                
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check valid term
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Decomposition failed at valid term ({i},{j},{k}): sum={s}")
                
                # Check invalid terms (optional, but good for sanity)
                # We know the construction ensures zeros elsewhere, but let's spot check one
                # Actually, the construction guarantees that for any (a,b,c) not matching (i,j,k),
                # the product is 0 because the indices don't align for any single r.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
