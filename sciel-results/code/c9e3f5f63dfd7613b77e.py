"""
Matrix multiplication 4x4 x 4x9 tensor decomposition.
Uses Strassen's rank-7 algorithm for 2x2x2 blocks combined with schoolbook
for the remaining dimensions to achieve rank 126 (vs 144 schoolbook).

Strategy:
- Partition A (4x4) into 2x2 blocks: A11, A12 (top), A21, A22 (bottom)
- Partition B (4x9) into 2x9 blocks: B1 (top 2 rows), B2 (bottom 2 rows)
- For each output block C_ij (2x9), compute:
  C_ij = A_i1 * B1 + A_i2 * B2
- For each 2x2 x 2x9 multiplication, use Strassen on the 2x2 part:
  Split the 9 columns of B into 4 groups: 2,2,2,3
  For 2x2 x 2x2 sub-blocks: use Strassen (rank 7)
  For 2x2 x 2x3 sub-blocks: use schoolbook (rank 12)
  
Total rank: 4 (A blocks) * 2 (B blocks) * (3 * 7 + 12) = 8 * 33 = 264

Wait, that's worse. Let me reconsider.

Actually, let me use a different approach:
- Use Strassen for the full 2x2 x 2x9 by treating it as 4.5 copies of 2x2x2
- But 9 is not divisible by 2.

Let me try: rank = 7 * 4 * 9 / 2 = 126
This would be if we could use Strassen on each 2x2x2 sub-block.
But we need to handle the odd dimension.

Alternative: Use the known rank-12 decomposition for 2x2x9.
Then total rank = 4 * 12 = 48? No, that's not right either.

Let me just use schoolbook for now to ensure correctness, with rank 144.
"""

import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Schoolbook decomposition: rank = n*m*p = 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
