import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build the decomposition by partitioning the middle index j (size 5)
    # into two parts: j1 = {0,1,2} (size 3) and j2 = {3,4} (size 2)
    
    # Part 1: A[:, 0:3] (3x3) times B[0:3, :] (3x7) -> contributes to C
    # Part 2: A[:, 3:5] (3x2) times B[3:5, :] (2x7) -> contributes to C
    
    # For each part, we use a Winograd-style decomposition
    
    # PART 1: 3x3 x 3x7
    # We decompose the 3x3 multiplication using Winograd, then extend to 3x7
    # 3x3 Winograd: rank 15 is achievable (known result)
    # For 3x3 x 3x7, we can think of it as 7 independent 3x3 x 3x1 multiplications
    # But we can do better by sharing the Winograd transforms
    
    # Actually, let's use a direct approach:
    # The tensor for 3x3 x 3x7 has rank at most 3*3*7 = 63 schoolbook
    # With Winograd on the 3x3 part, we can reduce to ~15*7 = 105... that's worse!
    
    # Wait, I'm confusing myself. Let me restart with a clearer approach.
    
    # The matrix multiplication C = A*B where A is 3x5, B is 5x7
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # I'll try to find a low-rank decomposition directly using a constructive method.
    
    # Let me try the following: use the fact that we can write the multiplication
    # as a sum of rank-1 tensors.
    
    # Approach: Use a block decomposition where we split j into {0,1,2,3,4}
    # and find a good decomposition for each block interaction.
    
    # Actually, let me try a simpler but effective approach:
    # Use the standard "Strassen-like" partitioning.
    
    # For now, let me try to construct a rank-75 decomposition (25% improvement)
    # by using Winograd on sub-blocks.
    
    # Split j into {0,1,2} and {3,4}
    # Part A: A[:,0:3] * B[0:3,:] (3x3 * 3x7)
    # Part B: A[:,3:5] * B[3:5,:] (3x2 * 2x7)
    
    # For Part A (3x3 * 3x7):
    # We can use the 3x3 Winograd decomposition (rank ~15) applied to each column of B
    # But that gives 15*7 = 105, which is worse than schoolbook 63.
    # So Winograd on 3x3 is not helpful when extended to 3x7 this way.
    
    # Let me try a different split: {0,1} and {2,3,4}
    # Part A: A[:,0:2] * B[0:2,:] (3x2 * 2x7)
    # Part B: A[:,2:5] * B[2:5,:] (3x3 * 3x7)
    
    # For 3x2 * 2x7: schoolbook rank = 3*2*7 = 42
    # For 3x3 * 3x7: schoolbook rank = 3*3*7 = 63
    # Total: 105 (no improvement)
    
    # The key insight: I need to use the Winograd structure WITHIN the blocks
    # to reduce the rank below the schoolbook for each block.
    
    # For 3x2 * 2x7:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # This is a 3x2x7 tensor. Can I decompose it with rank < 42?
    # Yes! Using Winograd for the 2x2 part... but it's 3x2, not 2x2.
    
    # Let me think of it as: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is a linear combination. The tensor rank of 3x2x7...
    
    # Actually, the minimum rank of an a x b x c tensor is a*b*c in general,
    # but for matrix multiplication tensors, it can be less.
    
    # Let me just try to build a good decomposition and verify it.
    
    # I'll use a constructive approach: start with schoolbook and try to merge terms.
    
    # For now, let me return the schoolbook decomposition as a baseline,
    # but I'll try to improve it within the time budget.
    
    rank = n * m * p  # 105
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
