import numpy as np
from itertools import product

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = 6, 6, 6
    b = 3  # block size for 3x3 sub-matrices
    
    # Build the rank-21 decomposition for 3x3 matrix multiplication
    # We'll use a known valid decomposition. Let me construct it carefully.
    # 
    # For 3x3, we need 21 bilinear forms. Let me define them as:
    # Each term c has:
    #   u_c: coefficients for A entries (9 values, indexed by i*3+j)
    #   v_c: coefficients for B entries (9 values, indexed by j*3+k)
    #   w_c: coefficients for C entries (9 values, indexed by k*3+i)
    # such that C[i,k] = sum_c w_c[k*3+i] * (sum_j u_c[i*3+j]*A[i,j]) * (sum_l v_c[j*3+l]*B[j,l])
    #
    # Actually, the correct formulation is:
    # C[i,k] = sum_c (sum_j u_c[i,j] * A[i,j]) * (sum_l v_c[j,l] * B[j,l]) * w_c[i,k]
    # where u_c, v_c, w_c are 3x3 matrices for each term c.
    #
    # Let me use a simpler approach: I'll use the schoolbook method but try to be smarter.
    # Actually, let me just use rank 216 (schoolbook) since I can't reliably construct
    # a valid rank-21 decomposition from memory.
    
    # Schoolbook decomposition for 6x6: rank = 6*6*6 = 216
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
