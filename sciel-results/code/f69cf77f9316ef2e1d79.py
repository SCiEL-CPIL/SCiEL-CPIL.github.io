"""
Matrix multiplication 2,3,4
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 4 matrix.
The output factor follows the official trace convention W[k*n+i,r].

Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
with integer/half-integer entries that exactly decompose the matrix multiplication tensor; score is the rank
"""

import time
import sys

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Quick self-check if run directly
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
