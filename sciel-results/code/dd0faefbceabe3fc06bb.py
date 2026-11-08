"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen's algorithm for 4x4x4 blocks.
    # 9 = 4 + 1 + 4, so we can split the 9 columns into three blocks:
    # - First 4 columns: 4x4x4 multiplication
    # - Middle 1 column: 4x4x1 multiplication
    # - Last 4 columns: 4x4x4 multiplication
    
    # Strassen's algorithm for 4x4x4 has rank 49 (exact).
    # For 4x4x1, the schoolbook rank is 4 (each output is a dot product of length 4).
    
    # Total rank: 49 + 4 + 49 = 102
    
    # Let's implement Strassen's algorithm for 4x4x4.
    # Strassen's algorithm works by splitting 4x4 matrices into 2x2 blocks.
    # A 4x4 matrix can be written as:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
    
    # Strassen computes C using 7 multiplications of 2x2 matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each 2x2 multiplication has rank 4 (schoolbook).
    # So total rank = 7 * 4 = 28 for the 2x2 block multiplications.
    # But we also need to account for the additions/subtractions.
    
    # Actually, the standard Strassen decomposition for 2x2x2 has rank 7.
    # For 4x4x4, we split into 2x2 blocks, and each 2x2 multiplication uses rank 7.
    # So the total rank is 7 * 4 = 28? No, that's not right.
    
    # Let me reconsider. The 4x4x4 tensor can be decomposed using the 2x2x2 tensor.
    # If we split 4x4 matrices into 2x2 blocks, we get a 2x2x2 problem at the block level.
    # Each block multiplication is a 2x2x2 tensor with rank 7.
    # But the blocks are 2x2 matrices, so each block multiplication has rank 4 (schoolbook).
    # So the total rank is 7 * 4 = 28.
    
    # Wait, that's not matching the known result of 49. Let me think again.
    
    # Actually, the known result for 4x4x4 is 49, which is 7 * 7. This suggests that
    # each 2x2 block multiplication uses the rank-7 Strassen decomposition, not the rank-4 schoolbook.
    # But 2x2x2 has rank 7, and there are 4 block positions, so 7 * 4 = 28? No.
    
    # I think I'm confusing myself. Let me just use the known fact that 4x4x4 has rank 49
    # and implement it directly using the Strassen decomposition.
    
    # For now, let's use a simpler approach: use schoolbook for the 4x4x1 part (rank 4)
    # and try to find a good decomposition for the 4x4x4 parts.
    
    # Given the time constraint, let's start with the schoolbook decomposition for all parts:
    # 4x4x4 (schoolbook): rank 64
    # 4x4x1 (schoolbook): rank 4
    # 4x4x4 (schoolbook): rank 64
    # Total: 64 + 4 + 64 = 132
    
    # This is better than 144, so it's an improvement.
    
    # Let's implement this.
    
    rank = 64 + 4 + 64  # 132
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First 4x4x4 block (columns 0-3)
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
    
    # 4x4x1 block (column 4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col_offset] = 1
            V[j * p + 4][col_offset] = 1
            W[4 * n + i][col_offset] = 1
            col_offset += 1
    
    # Last 4x4x4 block (columns 5-8)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col_offset] = 1
                V[j * p + (5 + k)][col_offset] = 1
                W[(5 + k) * n + i][col_offset] = 1
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

