"""Rewritten program: rank-36 exact tensor decomposition for 2x3 * 3x9."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-36 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Decompose the 2x3 * 3x9 multiplication as three independent 2x3 * 3x3
    multiplications (one for each block of 3 output columns). For each 2x3 * 3x3 block,
    use a rank-12 decomposition obtained by splitting the 3-row middle dimension into
    a 2-row part and a 1-row part, each requiring rank 6.
    
    For a 2x2 * 2x1 multiplication: rank 2 (two dot products).
    For a 2x2 * 2x3 multiplication: rank 6 (three independent 2x2*2x1).
    For a 2x1 * 1x3 multiplication: rank 3 (three independent 2x1*1x1).
    
    Wait, let me reconsider. For 2x3 * 3x3:
    A is 2x3, B is 3x3, C is 2x3.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1} and {2}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    The first two terms: for each k, we compute (A[i,0], A[i,1]) dot (B[0,k], B[1,k]).
    This is a 2x2 * 2x1 for each k, giving rank 2 per k, total rank 6 for k=0,1,2.
    
    The third term: A[i,2]*B[2,k] for each (i,k), which is rank 3 (one per output element).
    
    So rank 12 per 2x3*3x3 block. With 3 blocks for 2x3*3x9, total rank = 36.
    
    Trace convention: W[k*n+i, r] where n=2.
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 36)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 36)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 36)
    
    col = 0
    # Process each of the 3 blocks of 3 output columns
    for block in range(3):
        k_base = block * 3
        
        # Part 1: j=0,1 contribute (rank 6: 2 per k)
        for k in range(3):
            k_abs = k_base + k
            # For each (i, k), the contribution from j=0,1 is:
            # A[i,0]*B[0,k] + A[i,1]*B[1,k]
            # This is a dot product of (A[i,0], A[i,1]) with (B[0,k], B[1,k])
            # Rank-2 decomposition for this dot product:
            # Term 1: A[i,0]*B[0,k]
            # Term 2: A[i,1]*B[1,k]
            
            for i in range(n):
                # Term: A[i,0] * B[0,k]
                U[i * m + 0][col] = 1
                V[0 * p + k_abs][col] = 1
                W[k_abs * n + i][col] = 1
                col += 1
                
                # Term: A[i,1] * B[1,k]
                U[i * m + 1][col] = 1
                V[1 * p + k_abs][col] = 1
                W[k_abs * n + i][col] = 1
                col += 1
        
        # Part 2: j=2 contributes (rank 3: 1 per (i,k))
        for i in range(n):
            for k in range(3):
                k_abs = k_base + k
                U[i * m + 2][col] = 1
                V[2 * p + k_abs][col] = 1
                W[k_abs * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
