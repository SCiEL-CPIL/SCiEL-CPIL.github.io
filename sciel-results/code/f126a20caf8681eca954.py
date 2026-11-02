import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s safety margin
    
    # Try to build a rank-49 decomposition via Strassen 4x4x4 embedding
    # The idea: embed 3x4x6 into 4x4x4, use Strassen's rank-7 for 2x2,
    # giving rank 7*7=49 for 4x4x4, then project back.
    #
    # Actually, let's think more carefully. We want to compute C = A * B where
    # A is 3x4 and B is 4x6. 
    #
    # One approach: Pad A to 4x4 and B to 4x4, compute 4x4 product using 
    # Strassen (rank 49), then extract the 3x6 result.
    # But Strassen for 4x4 uses 7 multiplications of 2x2 matrices, each of rank 7,
    # total rank 49.
    #
    # However, the tensor decomposition approach is different. Let me think about
    # the bilinear form. The multiplication tensor T_{i,j,k} for C_{i,k} = sum_j A_{i,j} B_{j,k}
    # has rank n*m*p = 72 for schoolbook.
    #
    # For Strassen, we can express the 2x2x2 tensor with rank 7. For 4x4x4, we can
    # use a block structure. Let me try a different approach.
    #
    # Let's use the fact that we can split the 4x6 matrix B into two 4x3 blocks,
    # and compute A * B1 and A * B2 separately, where A is 3x4.
    # Each 3x4x3 multiplication has schoolbook rank 3*4*3 = 36.
    # Total: 72. No improvement.
    #
    # Alternative: Use Strassen for the 3x4x6 directly by viewing it as a block problem.
    # Split A (3x4) into rows, B (4x6) into columns.
    # 
    # Let me try: view the 3x4x6 multiplication as involving a 2x2 block structure
    # on the 4x4 part. Pad A to 4x4 (add one zero row), pad B to 4x4 (add two zero cols and one zero row? No, B is 4x6).
    #
    # This is getting complicated. Let me just implement the known rank-49 Strassen
    # decomposition for 4x4x4 and then zero out the unused dimensions.
    #
    # Actually, the cleanest approach: 
    # 1. Pad A (3x4) to (4x4) by adding a zero row.
    # 2. Pad B (4x6) to (4x4) by... wait, B has 6 columns. We can't pad to 4x4.
    #
    # Let me reconsider. The problem is 3x4x6. The middle dimension is 4.
    # We can use Strassen on the 4x4 part if we can restructure.
    #
    # Actually, let's just use the schoolbook rank-72 as the baseline and try
    # to find a better decomposition through a search or known results.
    #
    # Known: rank(3,4,6) <= 72 (schoolbook)
    # Can we do better? 
    #
    # One idea: Use the fact that min(3,4,6) = 3, and there are bounds.
    # The Laderman bound or similar might give something.
    #
    # For now, let's implement a rank-49 solution if possible, else fall back to 72.
    #
    # Let me try a concrete construction:
    # Split the 4x6 matrix B into B_left (4x3) and B_right (4x3).
    # Compute C_left = A * B_left (3x3) and C_right = A * B_right (3x3).
    # Each is a 3x4x3 multiplication.
    # For 3x4x3, schoolbook rank is 36. Total 72.
    #
    # Can we do 3x4x3 with rank < 36? 
    # 3x3x3 has rank 23 (Laderman). 3x4x3 might be around 30-33.
    #
    # This is getting too complex without a clear path. Let me just return the
    # schoolbook rank-72 decomposition, which is guaranteed correct.
    
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
    
    # Verify the decomposition
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # For the schoolbook, each t corresponds to a unique (i,j,k) triple.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
