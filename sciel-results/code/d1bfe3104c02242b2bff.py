"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Current best known: Rank 42 using Strassen for 6 pairs of columns.
    Potential improvement: Rank 30 using rank-10 decomposition for 2x2x4 (3 groups).
    Since the exact coefficients for the rank-10 decomposition are not trivial to recall 
    with 100% confidence, we implement the robust Rank 42 solution.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Number of column pairs
    num_pairs = p // 2  # 6 pairs
    
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U[0][c] = 1; U[3][c] = 1
                V[0*p + 2*t][c] = 1; V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1; W[(2*t+1)*n + 1][c] = 1
            elif s == 1:
                U[2][c] = 1; U[3][c] = 1
                V[0*p + 2*t][c] = 1
                W[2*t*n + 1][c] = 1; W[(2*t+1)*n + 1][c] = -1
            elif s == 2:
                U[0][c] = 1
                V[0*p + 2*t+1][c] = 1; V[1*p + 2*t+1][c] = -1
                W[(2*t+1)*n + 0][c] = 1; W[(2*t+1)*n + 1][c] = 1
            elif s == 3:
                U[3][c] = 1
                V[1*p + 2*t][c] = 1; V[0*p + 2*t][c] = -1
                W[2*t*n + 0][c] = 1; W[2*t*n + 1][c] = 1
            elif s == 4:
                U[0][c] = 1; U[1][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = -1; W[(2*t+1)*n + 0][c] = 1
            elif s == 5:
                U[2][c] = 1; U[0][c] = -1
                V[0*p + 2*t][c] = 1; V[0*p + 2*t+1][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 6:
                U[1][c] = 1; U[3][c] = -1
                V[1*p + 2*t][c] = 1; V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
