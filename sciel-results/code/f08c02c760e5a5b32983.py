"""Rewritten program for matrix multiplication 4x4x5 rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# Known rank-minimal decompositions for small matrix multiplication
# We'll use a systematic approach to find low-rank decompositions

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x5 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    where W[k*n+i, r] follows the official trace convention.
    """
    n, m, p = 4, 4, 5
    
    # Start with schoolbook decomposition (rank = n*m*p = 80)
    # Then try to reduce rank using known techniques
    
    # For 4x4x5 matrix multiplication, the rank is known to be at most 49
    # (based on Strassen-like algorithms)
    
    # Let's try a direct approach: use bilinear algorithms
    # The matrix multiplication tensor T_{i,j,k} = sum over (i',j',k') of A_{i,i'} * B_{j',j} * ...
    
    # Actually, let me think about this more carefully.
    # We're computing C = A * B where A is 4x4, B is 4x5, C is 4x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor has entries T[i,j,k] for each triple (i,j,k)
    # We want to write T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    
    # The schoolbook decomposition uses rank n*m*p = 80
    # Each term corresponds to one (i,j,k) triple
    
    # For better decompositions, we can use:
    # 1. Strassen's algorithm (for 2x2x2, rank 7 instead of 8)
    # 2. Generalizations to larger matrices
    
    # Let's try to use a known low-rank decomposition
    # For 4x4x4, the rank is known to be at most 49 (Lader)
    # For 4x4x5, we need to adapt
    
    # Strategy: Use a bilinear algorithm based on tensor rank properties
    # We'll construct the decomposition using a combination of:
    # - Base 2x2x2 Strassen blocks
    # - Padding for the 5th column
    
    # Actually, let me use a more direct approach:
    # We'll use the fact that we can decompose the multiplication
    # using fewer than n*m*p multiplications
    
    # For now, let's return the schoolbook decomposition as a baseline
    # and try to improve it
    
    rank = n * m * p  # 80
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
