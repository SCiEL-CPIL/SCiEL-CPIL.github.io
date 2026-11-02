"""Matrix multiplication 2x3x10 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of matrix multiplication.
    
    The decomposition is of the form:
    C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook algorithm:
    Rank is n*m*p.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, but numpy is allowed and might be faster for verification if needed,
    # but the output format expects 'U', 'V', 'W' which are factor matrices.
    # The seed returns lists of lists. I will stick to that or return numpy arrays if accepted.
    # The prompt says "factor matrices of shapes...". Numpy arrays are matrices.
    # Let's use numpy for efficiency in construction if we were doing search, but for fixed schoolbook, lists are fine.
    # However, returning numpy arrays is often safer for "matrix" type expectations.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                U[i * m + j, col] = 1
                # Index in V: row j*p + k
                V[j * p + k, col] = 1
                # Index in W: row k*n + i
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
