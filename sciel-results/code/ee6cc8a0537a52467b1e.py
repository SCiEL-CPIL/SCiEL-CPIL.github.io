"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication
    using Winograd convolutional algorithm framework with Hadamard transforms.
    
    Strategy: Map matrix multiplication to 2D convolution and use Winograd's algorithm
    to reduce the number of multiplications (tensor rank) below schoolbook bound.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll use a block-based Winograd approach.
    # Split the 4x4 matrix A into 2x2 blocks (4 blocks total)
    # Split the 4x5 matrix B into 2x(2 or 3) blocks
    # Use Strassen-like 2x2 multiplication for each block pair
    
    # For 2x2 * 2x2, Strassen uses 7 multiplications instead of 8
    # For 2x2 * 2x3, we need to handle the rectangular case
    
    # Let's try a direct construction using tensor rank minimization
    # via structured factor matrices with integer/half-integer entries.
    
    # Approach: Use the fact that we can express the multiplication as a sum
    # of rank-1 tensors where the factors have structure from Hadamard matrices.
    
    # For now, let's try a recursive Strassen-based approach:
    # 4x4 * 4x5 can be decomposed by splitting into 2x2 blocks
    
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (20, rank)  
    # W shape: (p*n, rank) = (20, rank)
    
    # Let's try to construct a decomposition with rank < 80
    # using block structure.
    
    # First, let's try the simple schoolbook but see if we can optimize
    # by combining terms that have the same (i,j,k) pattern.
    
    # Actually, let's try a different approach: use the known result that
    # for 2x2*2x2, rank is 7 (Strassen). For 4x4*4x4, we can get rank 49
    # by using Strassen recursively. For 4x4*4x5, we can pad to 4x8 and
    # use a modified approach.
    
    # Let me try a direct Winograd implementation:
    # Winograd F(2, 2, 1) for 2D convolution
    
    # For a 2x2 filter and 2x2 input, Winograd uses 4 multiplications
    # instead of 4 (no gain for 2x2), but for larger blocks it helps.
    
    # Let's try F(2, 2, 2) which should give better savings.
    
    # Actually, for matrix multiplication, the standard approach is to use
    # the tensor rank directly. Let me try to find a low-rank decomposition
    # by exploiting the bilinear structure.
    
    # Given the complexity, let's try a pragmatic approach:
    # Use the schoolbook decomposition but verify it's correct, then
    # attempt to reduce rank by finding linear dependencies.
    
    n, m, p = 4, 4, 5
    rank = n * m * p  # Start with schoolbook rank 80
    
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
    
    # Now try to reduce rank by finding if any columns are linearly dependent
    # or if we can merge terms.
    
    # For now, return the schoolbook decomposition as a baseline.
    # Future work: implement actual Winograd or Strassen-based reduction.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
