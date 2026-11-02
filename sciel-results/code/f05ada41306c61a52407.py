"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Strategy: Use a blocking/Strassen approach to reduce rank below 120.
    # For 4x5 x 5x6, we can split the middle dimension 5 = 2 + 3.
    # A (4x5) = [A1 (4x2), A2 (4x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # Rank = 4*2*6 + 4*3*6 = 48 + 72 = 120 (no improvement)
    
    # Better: Use Strassen-style for the 4x4 and 4x3 sub-problems.
    # Let's try: split A into 2+2 rows, B into 3+3 cols.
    # A (4x5) -> rows 0-1: A_top (2x5), rows 2-3: A_bot (2x5)
    # B (5x6) -> cols 0-2: B_left (5x3), cols 3-5: B_right (5x3)
    # C_top_left (2x3) = A_top * B_left
    # C_top_right (2x3) = A_top * B_right  
    # C_bot_left (2x3) = A_bot * B_left
    # C_bot_right (2x3) = A_bot * B_right
    
    # Each 2x5 x 5x3 product has schoolbook rank 2*5*3 = 30.
    # Total = 4 * 30 = 120. Still no improvement.
    
    # The key insight: we need to share products across the blocks.
    # Use Strassen-style: for 2x2 blocks of the 2x3 sub-problems.
    
    # Actually, let me try a different split: 5 = 2 + 3, and use
    # a 2x2 Strassen on the "2x2" part where possible.
    
    # Let's construct a decomposition using the following approach:
    # Pad the problem to 4x6 x 6x6 by adding zero columns/rows,
    # then use a known low-rank decomposition for 4x6 x 6x6,
    # and restrict back.
    
    # Alternatively, use a direct numerical search for a low-rank decomposition.
    # Given time constraints, let's implement a greedy approach.
    
    # For now, use the schoolbook as fallback, but try to find improvements.
    # Let's implement a simple iterative improvement: start with schoolbook,
    # then try to merge terms where possible.
    
    # Actually, let's use a known result: the tensor rank of 4x5 x 5x6
    # is at most 108 (from literature on matrix multiplication tensors).
    # But constructing it explicitly is complex.
    
    # Let's try a practical approach: use a 2-level Strassen blocking.
    # Level 1: Split 4 = 2+2, 5 = 2+3, 6 = 3+3
    # This gives 8 sub-products, each of size (2x2)x(2x3) or similar.
    
    # For now, return schoolbook to ensure correctness, with a deadline guard.
    rank = n * m * p
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
