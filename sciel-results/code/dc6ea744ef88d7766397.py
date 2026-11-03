"""
Constructs the factor matrices for the matrix multiplication tensor of dimensions 2x3x9.
Uses the schoolbook algorithm which yields a rank of 54.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    # Dimensions of the factor matrices
    N_U = n * m  # 2 * 3 = 6
    N_V = m * p  # 3 * 9 = 27
    N_W = p * n  # 9 * 2 = 18
    
    # Schoolbook rank: n * m * p
    R = n * m * p  # 54
    
    # Pre-allocate matrices
    U_list = [[0] * R for _ in range(N_U)]
    V_list = [[0] * R for _ in range(N_V)]
    W_list = [[0] * R for _ in range(N_W)]
    
    # Iterate through all rank components
    # For each (i, j, k), we create a rank-1 term
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U_list[u_idx][col_idx] = 1
                V_list[v_idx][col_idx] = 1
                W_list[w_idx][col_idx] = 1
                
                col_idx += 1
        
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
# EVOLVE-BLOCK-END
