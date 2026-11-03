"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Subspace embedding and null-space projection.
    We start from the schoolbook decomposition of the larger 4x4 * 4x8 tensor (rank 128).
    The 3x4 * 4x8 problem is a restriction where the 4th row of the first matrix is zero.
    
    The key insight: For the 4x4 * 4x8 tensor, the schoolbook decomposition has 128 terms
    indexed by (i, j, k) where i in {0,1,2,3}, j in {0,1,2,3}, k in {0,...,7}.
    
    For the 3x4 * 4x8 tensor, we only need i in {0,1,2}. The terms with i=3 are "extra"
    in the sense that they don't contribute to the 3x4 product when the 4th row of A is zero.
    
    However, we can't simply drop them. Instead, we observe that the 3x4 * 4x8 tensor is
    a sub-tensor of the 4x4 * 4x8 tensor. The minimal rank of the 3x4 * 4x8 tensor is
    at most the rank of the 4x4 * 4x8 tensor, but potentially lower.
    
    Actually, the schoolbook decomposition for 3x4 * 4x8 has rank 96 (3*4*8).
    The question is whether we can do better.
    
    For now, we return the schoolbook decomposition with rank 96.
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
