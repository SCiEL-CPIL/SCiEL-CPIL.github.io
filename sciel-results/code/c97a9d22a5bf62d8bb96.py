import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R = n * m * p  # 84 for 3x4x7

    # Initialize factor matrices
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)

    # We'll use a block-based approach:
    # A is 3x4, B is 4x7
    # Split A into left 3x2 (A_L) and right 3x2 (A_R)
    # Split B into top 2x7 (B_T) and bottom 2x7 (B_B)
    # C = A_L * B_T + A_R * B_B
    # Each product is 3x2 * 2x7 = 3x7
    # For each 3x2x7 product, we can use further decomposition.
    # For 3x2x7: split into 3x2x3 + 3x2x3 + 3x2x1
    # 3x2x3: schoolbook gives 18 multiplications
    # 3x2x1: schoolbook gives 6 multiplications
    # Total per 3x2x7: 18+18+6 = 42
    # Total: 42 + 42 = 84 (same as schoolbook)
    
    # Let's try a different approach: use Strassen-like decomposition for the 3x2x7 blocks
    # Actually, for 3x2, we can view it as 3x1 * 1x2 + 3x1 * 1x2
    # This doesn't help directly.
    
    # Let's just use schoolbook decomposition, which is guaranteed correct.
    # The rank is 84.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
