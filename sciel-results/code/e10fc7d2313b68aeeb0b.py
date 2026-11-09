import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # The rank 28 solution is constructed using a block-structured Strassen approach.
    # A (2x4) is split into two 2x2 blocks.
    # B (4x4) is split into two 2x4 blocks (top and bottom).
    # C = A_L B_T + A_R B_B.
    # Each term is a 2x2 by 2x4 multiplication.
    # The rank of 2x2 by 2x4 multiplication is 14 (7 for each 2x2 block of B, sharing A factors).
    # Total rank 14 + 14 = 28.
    
    U, V, W = _strassen_28(n, m, p)
    return {"U": U, "V": V, "W": W}

def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices: 0:00, 1:01, 2:10, 3:11
    # W indices: 0:00, 1:01, 2:10, 3:11
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    # Blocks for the 4x4 matrix B
    # B is 4x4. Split into B_T (2x4) and B_B (2x4).
    # B_T rows 0-1, B_B rows 2-3.
    # For B_T, j is 0-1. For B_B, j is 2-3.
    # k is 0-3 for both.
    
    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # A_L cols 0-1. A_R cols 2-3.
    # i is 0-1 for both.
    
    # We have two main parts: A_L * B_T and A_R * B_B.
    # Each part is a 2x2 * 2x4 multiplication.
    # This can be seen as two 2x2 * 2x2 multiplications sharing the A factors.
    # Let's iterate over the two A blocks (L, R) and two B blocks (T, B)?
    # No, A_L multiplies B_T. A_R multiplies B_B.
    # So we have two independent 2x2 * 2x4 multiplications.
    
    # Part 1: A_L (cols 0-1) * B_T (rows 0-1)
    # A_L indices: i in 0-1, j in 0-1.
    # B_T indices: j in 0-1, k in 0-3.
    # Result C indices: i in 0-1, k in 0-3.
    
    # Part 2: A_R (cols 2-3) * B_B (rows 2-3)
    # A_R indices: i in 0-1, j in 2-3.
    # B_B indices: j in 2-3, k in 0-3.
    # Result C indices: i in 0-1, k in 0-3.
    
    col = 0
    # Process Part 1: A_L * B_T
    # B_T is 2x4. Split into B_T1 (2x2, k=0-1) and B_T2 (2x2, k=2-3).
    # We compute A_L * B_T1 and A_L * B_T2.
    # Each uses 7 Strassen terms. Total 14 terms.
    
    for k_offset in [0, 2]:
        for term in strassen_terms:
            # U part (A_L)
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j # j is 0-1
                U[global_i * m + global_j][col] += coeff
            
            # V part (B_T)
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j # j is 0-1
                global_k = k + k_offset
                V[global_j * p + global_k][col] += coeff
            
            # W part (C)
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_offset
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    # Process Part 2: A_R * B_B
    # A_R is 2x2 (cols 2-3).
    # B_B is 2x4 (rows 2-3).
    # Split B_B into B_B1 (2x2, k=0-1) and B_B2 (2x2, k=2-3).
    
    for k_offset in [0, 2]:
        for term in strassen_terms:
            # U part (A_R)
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + 2 # j is 2-3
                U[global_i * m + global_j][col] += coeff
            
            # V part (B_B)
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + 2 # j is 2-3
                global_k = k + k_offset
                V[global_j * p + global_k][col] += coeff
            
            # W part (C)
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_offset
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1

    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return U, V, W
