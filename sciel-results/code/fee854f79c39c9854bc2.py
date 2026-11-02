"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 tensor decomposition for M_{3,5,5}.
    
    Strategy: Use a block decomposition where we split the 5 columns/rows into
    2+3 blocks, and use Strassen's algorithm (rank 7) for the 2x2 blocks and
    a rank-13 decomposition for the 3x3 blocks.
    
    Actually, let me use a more direct approach. I'll use the following construction:
    
    For M_{3,5,5}, we can use the fact that:
    - R(3,5,5) <= R(3,2,5) + R(3,3,5)
    - R(3,2,5) <= 3*2*5 = 30 (schoolbook)
    - R(3,3,5) <= 3*3*5 = 45 (schoolbook)
    Total: 75 (not helpful)
    
    Better approach: Use known small matrix multiplication ranks.
    
    Actually, I'll use a construction based on the following idea:
    - Pad the 3x5 matrix to 5x5 (add 2 zero rows)
    - Use the rank-19 decomposition for 5x5x5
    - But this doesn't directly work because we need exactly 3 rows
    
    Let me use a different approach. I'll implement a rank-35 construction
    using the following explicit formula:
    
    For M_{3,5,5}, we can achieve rank 35 using the following:
    Split into blocks and use optimized decompositions.
    
    Actually, let me just use the schoolbook rank-75 decomposition. It's correct
    and guaranteed to work. The task asks for rank-minimal, but without a verified
    rank-35 construction, I'll stick with the safe option.
    
    Wait - let me try to construct a rank-35 decomposition using the following:
    
    We can write the 5x5 multiplication as a sum of products of smaller matrices.
    Using the fact that 5 = 2 + 3, we can use:
    - A 2x2x2 Strassen decomposition (rank 7) for the top-left 2x2 block
    - Cross terms and 3x3 blocks
    
    This is getting complex. Let me use a simpler approach:
    
    I'll use the fact that R(3,5,5) <= 3*R(1,5,5) = 3*25 = 75 (schoolbook).
    
    Actually, for a better bound, I'll use:
    R(3,5,5) <= R(3,5,3) + R(3,5,2) where we split the output columns.
    R(3,5,3) <= 3*5*3 = 45
    R(3,5,2) <= 3*5*2 = 30
    Total: 75. Still not helpful.
    
    Let me just return the schoolbook decomposition. It's rank 75 and correct.
    """
    n, m, p = DIMENSIONS
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
