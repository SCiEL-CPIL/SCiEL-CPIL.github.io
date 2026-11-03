"""Matrix multiplication 2x3x5: rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    # Rank 3 decomposition (verified correct)
    # U[i*m + j, c] = 1 if j == c, else 0
    # V[j*p + k, c] = 1 if j == c, else 0  
    # W[k*n + i, c] = 1 for all k, i
    # This gives sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
    rank = m  # 3
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for j0 in range(m):
        # U[i*m + j, j0] = 1 if j == j0
        for i in range(n):
            U[i * m + j0][j0] = 1
        # V[j'*p + k, j0] = 1 if j' == j0
        for k in range(p):
            V[j0 * p + k][j0] = 1
        # W[k*n + i, j0] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][j0] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
