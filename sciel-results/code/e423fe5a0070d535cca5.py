import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns a block-decomposed decomposition for matrix multiplication 3x4 by 4x7.
    Splits the problem into 3x4x3 and 3x4x4 sub-problems.
    Currently uses schoolbook for sub-problems, resulting in rank 84.
    Structure allows for future replacement of sub-problems with lower-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    
    # Split p into p1=3 and p2=4
    p1 = 3
    p2 = p - p1 # 4
    
    # Total rank is sum of ranks of sub-problems
    # Sub-problem 1: 3x4 by 4x3 -> 3x3
    # Sub-problem 2: 3x4 by 4x4 -> 3x4
    
    rank1 = n * m * p1 # 3*4*3 = 36
    rank2 = n * m * p2 # 3*4*4 = 48
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col_offset = 0
    
    # --- Sub-problem 1: A (3x4) * B1 (4x3) -> C1 (3x3) ---
    # B1 corresponds to columns 0..2 of B
    # C1 corresponds to columns 0..2 of C
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                # U index: i*m + j
                # V index: j*p + k (k is 0..2, so j*7 + k)
                # W index: k*n + i (k is 0..2, so k*3 + i)
                
                U[i * m + j, col_offset] = 1
                V[j * p + k, col_offset] = 1
                W[k * n + i, col_offset] = 1
                col_offset += 1
                
    # --- Sub-problem 2: A (3x4) * B2 (4x4) -> C2 (3x4) ---
    # B2 corresponds to columns 3..6 of B
    # C2 corresponds to columns 3..6 of C
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):
                # U index: i*m + j
                # V index: j*p + k (k is 3..6, so j*7 + k)
                # W index: k*n + i (k is 3..6, so k*3 + i)
                
                U[i * m + j, col_offset] = 1
                V[j * p + k, col_offset] = 1
                W[k * n + i, col_offset] = 1
                col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
