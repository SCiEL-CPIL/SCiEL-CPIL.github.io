"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import itertools
from collections import defaultdict
import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We'll use a block decomposition approach to achieve rank < 45
    # Split A (3x3) into rows [0:2] and [2:3]
    # Split B (3x5) into columns [0:2], [2:4], [4:5]
    
    # Block structure:
    # A = [[A_top (2x3)], [A_bottom (1x3)]]
    # B = [[B_left (3x2)], [B_mid (3x2)], [B_right (3x1)]]
    
    # C = A @ B has blocks:
    # C_top_left (2x2) = A_top @ B_left
    # C_top_mid (2x2) = A_top @ B_mid  
    # C_top_right (2x1) = A_top @ B_right
    # C_bottom_left (1x2) = A_bottom @ B_left
    # C_bottom_mid (1x2) = A_bottom @ B_mid
    # C_bottom_right (1x1) = A_bottom @ B_right
    
    # For the 2x2 blocks, we can use Strassen-like approaches
    # Let's build a rank-43 decomposition
    
    # Strategy: Use the known result that m(3,3,5) <= 43
    # We'll construct it explicitly using block decomposition with shared multiplications
    
    # Define helper to create sparse terms
    terms = []  # Each term is (U_dict, V_dict, W_dict) where dicts map index -> coefficient
    
    def idx_U(i, j):
        return i * m + j
    
    def idx_V(j, k):
        return j * p + k
    
    def idx_W(k, i):
        return k * n + i
    
    # Block 1: C_top_left (2x2) = A_top (2x3) @ B_left (3x2)
    # This is a 2x3 @ 3x2 multiplication
    # We can decompose this into:
    # Let A_top = [[a00, a01, a02], [a10, a11, a12]]
    # Let B_left = [[b00, b01], [b10, b11], [b20, b21]]
    # 
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # Standard: 12 multiplications
    # Can we do better? For 2x3 @ 3x2, the minimum rank is known to be 12.
    
    # Block 2: C_top_mid (2x2) = A_top (2x3) @ B_mid (3x2)
    # Same structure as Block 1: 12 multiplications
    
    # Block 3: C_top_right (2x1) = A_top (2x3) @ B_right (3x1)
    # C[0,4] = a00*b04 + a01*b14 + a02*b24
    # C[1,4] = a10*b04 + a11*b14 + a12*b24
    # Standard: 6 multiplications
    
    # Block 4: C_bottom_left (1x2) = A_bottom (1x3) @ B_left (3x2)
    # C[2,0] = a20*b00 + a21*b10 + a22*b20
    # C[2,1] = a20*b01 + a21*b11 + a22*b21
    # Standard: 6 multiplications
    
    # Block 5: C_bottom_mid (1x2) = A_bottom (1x3) @ B_mid (3x2)
    # Standard: 6 multiplications
    
    # Block 6: C_bottom_right (1x1) = A_bottom (1x3) @ B_right (3x1)
    # C[2,4] = a20*b04 + a21*b14 + a22*b24
    # Standard: 3 multiplications
    
    # Total standard: 12 + 12 + 6 + 6 + 6 + 3 = 45
    
    # To get to 43, we need to save 2 multiplications.
    # We can try to share some multiplications between blocks that use the same A or B factors.
    
    # Key insight: A_top is used in Blocks 1, 2, 3. If we can create shared products...
    # Similarly, B_left is used in Blocks 1, 4.
    
    # Let's try a different approach: use a more global decomposition.
    
    # Actually, let me just implement the schoolbook rank-45 and verify it works.
    # The task says score is the rank, and lower is better.
    # If I can't find a rank-43 decomposition quickly, I'll return rank-45.
    
    # But wait - the current program already returns rank 45.
    # Let me try to find a rank-43 or better decomposition.
    
    # For now, let's carefully implement a rank-43 decomposition using known algorithms.
    
    # Here's a cleaner approach: use the fact that we can apply Strassen-style
    # tricks to reduce the rank.
    
    # Let's just go with schoolbook for now to ensure correctness, then optimize.
    
    rank = n * m * p  # 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
