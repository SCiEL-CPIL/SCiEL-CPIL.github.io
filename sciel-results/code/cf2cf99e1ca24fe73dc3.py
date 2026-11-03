"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Use a Strassen-based decomposition for 3x3 * 3x8
    # We can partition the 3x3 matrix A into 2x2 and 1x2 blocks
    # and use Strassen's algorithm for the 2x2 * 2x8 part,
    # plus direct multiplication for the remaining parts.
    
    # Strategy:
    # A is 3x3, B is 3x8, C is 3x8
    # Split rows of A: top 2 rows (2x3) and bottom 1 row (1x3)
    # Split columns of B: left 2 cols (3x2) and right 1 col (3x1)
    # Actually, let's split differently:
    
    # Better approach: Use 2x2 Strassen blocks
    # Split A into four 2x2 blocks? No, A is 3x3.
    
    # Let's use a different approach:
    # View 3x3 * 3x8 as:
    # - 2x3 * 3x8 (top 2 rows of A) -> 2x8 result
    # - 1x3 * 3x8 (bottom row of A) -> 1x8 result
    
    # For 2x3 * 3x8:
    # Split 2x3 into 2x2 + 2x1, and 3x8 into 2x8 + 1x8
    # This gets complicated. Let's use a known decomposition.
    
    # Actually, let's try a simpler approach:
    # Use the fact that 3x3 matrix multiplication has rank 7 (Strassen-like)
    # and we're multiplying by a 3x8 matrix.
    
    # Let's construct a decomposition with rank 21 or lower.
    
    # For now, let's return a valid schoolbook decomposition with rank 72
    # to ensure correctness, then try to improve.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
