import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 3x7 matrix B into three parts:
    # B1: 3x3 (cols 0-2)
    # B2: 3x3 (cols 3-5)
    # B3: 3x1 (col 6)
    # The multiplication C = A * B is split into C1 = A*B1, C2 = A*B2, C3 = A*B3.
    # The rank is the sum of the ranks of these sub-multiplications.
    # For 3x3x3, naive rank is 27.
    # For 3x3x1, naive rank is 9.
    # Total rank = 27 + 27 + 9 = 63.
    
    R = 27 + 27 + 9
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    
    # Helper to add a naive decomposition for a block of columns
    def add_naive_block(start_col, num_cols):
        nonlocal r
        for i in range(n):
            for j in range(m):
                for k in range(num_cols):
                    # Global column index
                    k_global = start_col + k
                    
                    # Index in U (A matrix): row i, col j -> i * m + j
                    a = i * m + j
                    # Index in V (B matrix): row j, col k_global -> j * p + k_global
                    b = j * p + k_global
                    # Index in W (C matrix): row k_global, col i -> k_global * n + i
                    c = k_global * n + i
                    
                    U[a, r] = 1
                    V[b, r] = 1
                    W[c, r] = 1
                    r += 1
    
    # Block 1: Cols 0-2 (3x3)
    add_naive_block(0, 3)
    
    # Block 2: Cols 3-5 (3x3)
    add_naive_block(3, 3)
    
    # Block 3: Col 6 (3x1)
    add_naive_block(6, 1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
