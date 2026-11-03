"""Program to find a low-rank decomposition for 2x3 * 3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    rank = n * m * p  # 42
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The term in C[i,k] is A[i,j] * B[j,k]
                # U corresponds to A indices: i*m + j
                # V corresponds to B indices: j*p + k
                # W corresponds to C indices: k*n + i (trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
