"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Constructs a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Total number of bilinear terms in the schoolbook algorithm
    total_terms = n * m * p  # 96
    
    # Theoretical lower bound for 3x4x8 is 41.
    # We will attempt to find a decomposition with rank R = 40 (close to lower bound).
    # If that fails, we fall back to higher ranks.
    
    # Strategy:
    # 1. Try to find a rank-40 decomposition using a heuristic/numerical approach.
    # 2. If not found, return a valid higher-rank decomposition.
    
    # For now, let's return a valid decomposition with rank 48.
    # This is achievable by using a rank-24 decomposition for 3x4x4 (if it exists)
    # or by a direct construction.
    
    # Let's try to construct a rank-48 decomposition directly.
    # We can split the 3x4x8 problem into two 3x4x4 problems.
    # For each 3x4x4, we use a rank-24 decomposition.
    
    # Since we don't have a known rank-24 decomposition for 3x4x4,
    # we will use a simpler approach: return a rank-48 decomposition
    # by combining pairs of schoolbook terms.
    
    # Actually, let's just return the schoolbook decomposition for now.
    # It's guaranteed to be valid, and we can optimize later.
    
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
