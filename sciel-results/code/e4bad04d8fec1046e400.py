"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 125 for 5x5x5
    
    # Initialize matrices with zeros
    # Using lists of lists as per interface requirement, though numpy is allowed, 
    # the return type must be {'U', 'V', 'W'} with entries.
    # The prompt says "integer/half-integer entries". 1.0 is an integer.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Construct the schoolbook decomposition
    # The tensor for matrix multiplication is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # In the index convention W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]:
    # For each term in the sum, we assign a unique column 'c'
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1.0
                # V index: j*p + k
                V[j * p + k][col] = 1.0
                # W index: k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
