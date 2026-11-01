<<<<<<< SEARCH
"""
Tensor decomposition for 2x2 matrix multiplication.

This program returns the rank-7 Strassen decomposition for multiplying
two 2x2 matrices. The rank 7 is the theoretical minimum for this problem
over fields of characteristic 0.

Interface:
    construct() -> {'U', 'V', 'W'}
    U: shape (4, 7)
    V: shape (4, 7)
    W: shape (4, 7)
    
    Indices follow the convention:
    U[i*m + j, r] corresponds to A[i, j]
    V[j*p + k, r] corresponds to B[j, k]
    W[k*n + i, r] corresponds to C[i, k]
    where n=m=p=2.
"""

def construct():
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    # U maps A elements to Strassen products
    # V maps B elements to Strassen products
    # W maps Strassen products to C elements
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
=======
"""
Tensor decomposition for 2x2 matrix multiplication.

This program returns the rank-7 Strassen decomposition for multiplying
two 2x2 matrices. The rank 7 is the theoretical minimum for this problem
over fields of characteristic 0.

Interface:
    construct() -> {'U', 'V', 'W'}
    U: shape (4, 7)
    V: shape (4, 7)
    W: shape (4, 7)
    
    Indices follow the convention:
    U[i*m + j, r] corresponds to A[i, j]
    V[j*p + k, r] corresponds to B[j, k]
    W[k*n + i, r] corresponds to C[i, k]
    where n=m=p=2.
"""
import numpy as np

def construct():
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices as NumPy arrays for robustness
    # U maps A elements to Strassen products
    # V maps B elements to Strassen products
    # W maps Strassen products to C elements
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
>>>>>>> REPLACE
