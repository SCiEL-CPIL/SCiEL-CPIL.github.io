"""Tensor decomposition for 4x4x6 matrix multiplication with reduced rank."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-reduced decomposition
    # Start with schoolbook rank = 96
    # Use Strassen-inspired approach for better rank
    
    # For 4x4x6, we can use the following approach:
    # Split the 6 columns into 4+2
    # Use Strassen for 4x4x4 (rank 49)
    # Add terms for the remaining 2 columns
    
    # Strassen's 2x2x2 rank-7 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # For 4x4x4 using 2x2 blocks, we get rank 7*4 = 28? No, that's not right.
    # Actually, for 4x4x4, the Strassen decomposition gives rank 49.
    
    # Let me implement the known best decomposition for 4x4x4 with rank 49.
    # This is the standard Strassen-based decomposition.
    
    # The 4x4x4 tensor can be decomposed using 49 rank-1 tensors.
    # I'll implement this explicitly.
    
    # Actually, let me use a different approach: construct the full 4x4x6 tensor
    # and try to find a lower rank decomposition by exploiting structure.
    
    # The matrix multiplication tensor T[i,j,k] = 1 if i is row, j is col of A, k is col of B
    # For C = AB where A is 4x4, B is 4x6, C is 4x6
    
    # Let me build the factor matrices directly using a known good decomposition.
    
    # I'll use the approach of decomposing into:
    # 1. 4x4x4 part using Strassen (rank 49)
    # 2. 4x4x2 part using schoolbook (rank 32)
    # Total: 81, which is worse than 96? No, 49+32=81 < 96
    
    # Wait, that's better! Let me implement this.
    
    # For the 4x4x4 part (columns 0-3), use Strassen rank 49
    # For the 4x4x2 part (columns 4-5), use schoolbook rank 32
    # Total rank: 81
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]  # 16 x 81
    V = [[0] * rank for _ in range(m * p)]  # 24 x 81
    W = [[0] * rank for _ in range(p * n)]  # 24 x 81
    
    # First 49 terms: Strassen for 4x4x4 (output columns 0-3)
    col = 0
    # Strassen decomposition for 4x4x4
    # Using 2x2 block structure
    # Each 2x2 block multiplication uses 7 terms
    # We have 4 such multiplications (for each 2x2 block of output)
    # But there's overlap, so total is 49, not 28
    
    # Let me implement the standard 4x4x4 rank-49 decomposition.
    # This is complex, so let me use a simpler approach:
    # Use schoolbook for the whole thing but try to find dependencies.
    
    # Actually, let me just use the schoolbook decomposition for now
    # and verify it's correct, then try to improve.
    
    # For now, return schoolbook to ensure correctness
    rank = n * m * p  # 96
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
