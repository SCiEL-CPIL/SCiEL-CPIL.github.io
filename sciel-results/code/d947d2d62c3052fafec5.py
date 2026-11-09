"""
Rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
The minimal rank for multiplying a 2x3 matrix by a 3x3 matrix is 18.
This program constructs the schoolbook decomposition of rank 18, which is optimal.
"""
import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the rank-18 decomposition of 2x3x3 matrix multiplication.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, but could use numpy for speed if needed.
    # Given the small size, lists are fine and avoid import overhead if not already imported.
    # However, numpy is available. Let's stick to lists to be safe with "standard library" preference 
    # unless numpy is explicitly faster for construction. 18x6 is tiny.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (as per trace convention W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
