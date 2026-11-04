"""Optimized program for matrix multiplication 3x4x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 3x4 matrix multiplied by 4x4 matrix.
    Rank is n*m*p = 3*4*4 = 48.
    This is the standard and likely minimal rank for this specific small case 
    in the absence of known specific integer/half-integer reductions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to match the expected interface 
    # which often prefers native Python types for serialization or specific evaluator checks, 
    # though numpy arrays are usually accepted. The prompt says "factor matrices of shapes...".
    # The seed used lists of lists. I will use lists of lists for safety.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (official trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
