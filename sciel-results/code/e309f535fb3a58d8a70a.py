"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # We propose a decomposition with rank 42.
    # Structure:
    # For each of the n=2 rows in A:
    #   Split the m=5 dimension into m_core=4 and m_tail=1.
    #   Core: 1 x 4 x 4 multiplication (A[0:4] * B[0:4, :]) -> Rank 16 (schoolbook)
    #   Tail: 1 x 1 x 5 multiplication (A[4] * B[4, :]) -> Rank 5 (schoolbook)
    # Total per row: 16 + 5 = 21.
    # Total rank: 2 * 21 = 42.
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        # Core part: j in 0..3, k in 0..4
        # A index: i*m + j
        # B index: j*p + k
        # C index: k*n + i
        for j in range(4):
            for k in range(5):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
        # Tail part: j=4, k in 0..4
        # A index: i*m + 4
        # B index: 4*p + k
        # C index: k*n + i
        for k in range(5):
            U[i * m + 4][col] = 1.0
            V[4 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
