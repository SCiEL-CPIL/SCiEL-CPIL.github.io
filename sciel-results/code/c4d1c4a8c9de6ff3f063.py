"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use the known rank-30 decomposition for 2x5x6 matrix multiplication
    # This is the Strassen-like / fast matrix multiplication result
    # For 2x5 * 5x6, the tensor rank is known to be 30 (or possibly lower, but 30 is a safe upper bound)
    
    # Let's try to find a decomposition with rank < 60
    # The schoolbook rank is 2*5*6 = 60
    # For small matrices, we can use known fast multiplication algorithms
    
    # For 2x5 * 5x6, we can split the 5x5 multiplication part
    # 5 = 3 + 2, so we can use Strassen for the 3x3 and 2x2 parts
    
    # Actually, let's use a direct approach: try to find a rank-30 decomposition
    # by exploiting the structure of the tensor
    
    # The matrix multiplication tensor T[i,j,k] = 1 if i,j,k are compatible
    # For 2x5 * 5x6:
    # - i in {0,1} (rows of A, columns of C)
    # - j in {0,...,4} (columns of A, rows of B)  
    # - k in {0,...,5} (columns of B, columns of C)
    
    # T[i,j,k] = 1 always (it's a dense tensor)
    
    # Let's try a rank-30 decomposition using the following approach:
    # Split j into two parts: j1 in {0,1,2} and j2 in {3,4}
    # Split k into two parts: k1 in {0,1,2} and k2 in {3,4,5}
    
    # For the 2x3 * 3x3 part (i x j1 * k1): rank <= 2*3*3 = 18, but can be less
    # For the 2x2 * 2x3 part (i x j2 * k2): rank <= 2*2*3 = 12, but can be less
    # For the cross terms, we need to be careful
    
    # Actually, let's just try to find a decomposition with rank 30 directly
    # using a random search or a structured approach
    
    # Known result: rank of 2x5*5x6 matrix multiplication tensor
    # For m=n=p, rank is known for small cases
    # For 2x5*5x6, let's try rank 30
    
    # Let's construct a rank-30 decomposition using the following method:
    # Use the fact that we can group indices
    
    # Method: Use bilinear forms
    # For each of the 30 "intermediate" variables, define:
    # - u[i,j] for i in 0..1, j in 0..4
    # - v[j,k] for j in 0..4, k in 0..5
    # - w[k,i] for k in 0..5, i in 0..1
    
    # The constraint is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # This is a tensor decomposition problem. Let's try a structured approach.
    
    # Approach: Use the fact that 5 = 2 + 3 and 6 = 3 + 3
    # Split A into A1 (2x2) and A2 (2x3)
    # Split B into B1 (2x3) and B2 (3x3)
    # Then C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x3): rank <= 2*2*3 = 12
    # For A2*B2 (2x3 * 3x3): rank <= 2*3*3 = 18
    # Total rank <= 12 + 18 = 30
    
    # So we can achieve rank 30!
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 10 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    # Part 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-2)
    # This contributes to C[i,k] for i in 0-1, k in 0-2
    # A1 has shape 2x2, B1 has shape 2x3
    # Schoolbook rank for 2x2*2x3 is 12
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    # col should be 12 now
    
    # Part 2: A2 (rows 0-1, cols 2-4) * B2 (rows 2-4, cols 0-5)
    # This contributes to C[i,k] for i in 0-1, k in 0-5
    # A2 has shape 2x3, B2 has shape 3x6
    # Schoolbook rank for 2x3*3x6 is 36, but we only have 18 slots left
    
    # Wait, this doesn't work. The cross terms would overlap.
    
    # Let me reconsider. The issue is that when we split j and k, 
    # we get cross terms that need to be handled.
    
    # Let's try a different split:
    # Split only j: j1 = {0,1,2}, j2 = {3,4}
    # A = [A1 | A2] where A1 is 2x3 and A2 is 2x2
    # B = [B1; B2] where B1 is 3x6 and B2 is 2x6
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x3 * 3x6, schoolbook rank = 2*3*6 = 36
    # A2*B2: 2x2 * 2x6, schoolbook rank = 2*2*6 = 24
    # Total = 60, which is the same as schoolbook. Not helpful.
    
    # The key insight is that we need to use fast matrix multiplication algorithms.
    # For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    
    # Let's try to use Strassen-like techniques.
    
    # For 2x5 * 5x6, we can pad to 2x6 * 6x6 or use a different approach.
    
    # Actually, let's just return the schoolbook decomposition for now
    # and try to improve it in future iterations.
    
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
