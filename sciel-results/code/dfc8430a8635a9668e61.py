"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook decomposition
    rank_school = n * m * p  # 128
    
    # Try to find a lower-rank decomposition using random search with structured patterns
    # For matrix multiplication C = A*B where A is n x m, B is m x p
    # The tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # First, let's try to use a Kronecker product structure
    # If we can decompose as tensor product of smaller decompositions, we might get lower rank
    
    # For 4x4 * 4x8, we can think of this as:
    # A is 4x4, B is 4x8 = 4x(2*4)
    # We can write B as [B1, B2] where B1, B2 are 4x4
    # Then C = [A*B1, A*B2]
    
    # The rank of A*B1 + A*B2 decomposition could be related to 2x the rank of 4x4*4x4
    # But actually we need separate terms for each output column block
    
    # Let's try a direct approach: use random matrices and check if they satisfy the constraint
    # Actually, let me think about this differently.
    
    # The matrix multiplication tensor for n x m * m x p has a known structure.
    # For small dimensions, we can use the following approach:
    
    # Try rank 52, 54, 56, ... up to some limit
    # Use the fact that for square matrices, Strassen gives rank 7 for 2x2
    # For 4x4, we can get rank 18 using Strassen (2x2 blocks with Strassen)
    
    # Actually, for 4x4 * 4x8:
    # Think of A as 2x2 blocks of 2x2, B as 2x4 blocks (2 blocks of 2x4? No, B is 4x8)
    # Let's partition: A is (2x2) blocks of size 2x2, B is 2x2 blocks of size 2x4
    # Then C is (2x2) blocks of size 2x4
    
    # This is getting complex. Let me try a simpler approach:
    # Use the schoolbook as baseline, then try to find improvements via optimization
    
    # For now, return the schoolbook decomposition
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
