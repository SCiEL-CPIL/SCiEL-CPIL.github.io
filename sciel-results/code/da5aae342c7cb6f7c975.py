"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We use a block decomposition where A is 4x4, B is 4x6, C is 4x6.
    # We treat this as a block matrix multiplication.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each block is 2x2
    
    # Actually, let's just use the known minimal rank for small dimensions if we can find one,
    # or use a known decomposition like Winograd for certain sizes.
    # For 4x4x6, the schoolbook rank is 96.
    # We can try to reduce rank using block Strassen-like methods.
    
    # Let's construct a rank-77 decomposition using block Strassen for the 2x2 blocks of A and B.
    # A is 4x4, so 2x2 blocks of 2x2.
    # B is 4x6, so 2x3 blocks of 2x2.
    
    # Let's define the tensor decomposition for the product of two 2x2 matrices (rank 7 via Strassen)
    # and combine them.
    
    # Actually, a simpler approach:
    # The task is to find a rank-minimal exact tensor decomposition.
    # For 4x4x6, the rank is at most 4*4*6 = 96.
    # We can try to use a known lower rank decomposition.
    
    # Let's use the fact that we can decompose the 4x4x6 multiplication into smaller parts.
    # We'll use a block decomposition where we multiply 2x2 blocks.
    
    # For now, let's use the schoolbook decomposition to ensure correctness, 
    # but we'll try to optimize it slightly if possible.
    
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
