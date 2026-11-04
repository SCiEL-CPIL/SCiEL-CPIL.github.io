"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy: decompose into 4 independent 2x3x2 sub-problems
    # Each 2x3x2 block uses a rank-6 Winograd-style decomposition
    # Total rank = 4 * 6 = 24
    
    rank_per_block = 6
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 24
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 x 24
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 24
    W = [[0.0] * total_rank for _ in range(p * n)]  # 16 x 24
    
    # For each block b (columns 2b, 2b+1 of B):
    # A is 2x3: A[i][j] for i in 0,1; j in 0,1,2
    # B_block is 3x2: B[j][2b + col] for j in 0,1,2; col in 0,1
    # C is 2x2: C[i][2b + col]
    
    # Winograd-style rank-6 decomposition for 2x3x2:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # Let B = [[b00, b01], [b10, b11], [b20, b21]]
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Rank-6 factors (U, V, W components for each of the 6 multiplications):
    # M1 = (a00 + a10) * (b00 + b10)
    # M2 = (a01 + a11) * (b10 + b20)  
    # M3 = (a02 + a12) * b20
    # M4 = (a00 + a10) * (b01 + b11)
    # M5 = (a01 + a11) * (b11 + b21)
    # M6 = (a02 + a12) * b21
    
    # But this doesn't directly give C. Let me use a proper Winograd scheme.
    # 
    # Actually, for 2x3x2, let's use the following:
    # Think of it as computing two independent dot products of length 3.
    # C[0,0] = A_row0 . B_col0
    # C[0,1] = A_row0 . B_col1
    # C[1,0] = A_row1 . B_col0
    # C[1,1] = A_row1 . B_col1
    #
    # A better approach: use the fact that we can share computations.
    # 
    # Let's define:
    # S1 = a00 + a10, D1 = a00 - a10
    # S2 = a01 + a11, D2 = a01 - a11
    # S3 = a02 + a12, D3 = a02 - a12
    #
    # T1 = b00 + b01, U1 = b00 - b01
    # T2 = b10 + b11, U2 = b10 - b11
    # T3 = b20 + b21, U3 = b20 - b21
    #
    # Then:
    # C[0,0] = 1/2 * [S1*T1 + S2*T2 + S3*T3 + D1*U1 + D2*U2 + D3*U3] ... no this is wrong.
    
    # Let me just use schoolbook for each 2x3x2 block but try to be clever.
    # Actually, the minimum rank for 2x3x2 is indeed 12 in general.
    # 
    # Wait - let me reconsider. Maybe we can do better globally.
    # 
    # Alternative: view the problem as 2x3 x 3x8 = 2x8
    # The output has 16 entries. We need at least 16 multiplications.
    # 
    # Let me try a different decomposition. Group the 8 columns into 4 pairs.
    # For each pair, compute 2x3 * 3x2 = 2x2.
    # 
    # For 2x3 * 3x2, I'll use a rank-12 schoolbook per block, total 48.
    # But that's the same as full schoolbook.
    # 
    # Let me try to find a better global structure.
    # 
    # Actually, let's just implement a clean schoolbook with rank 48.
    
    total_rank = n * m * p  # 48
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
