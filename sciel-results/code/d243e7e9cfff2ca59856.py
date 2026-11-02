"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Known lower bound for matrix multiplication rank is at least n*m or m*p etc.
    # For 3x3 * 3x7, schoolbook is 63.
    # Strassen reduces 2x2x2 from 8 to 7.
    # For 3x3x3, best known is 23 (Landsberg, etc.).
    # For 3x3x7, we can potentially do better than 63.
    
    # Let's try to find a decomposition with rank < 63.
    # We can use a randomized search with gradient-free methods or integer programming.
    # Since time is limited (150s), we'll use a heuristic approach.
    
    # Strategy: Start with schoolbook, then try to merge terms using Strassen-like ideas.
    # Or use a randomized tensor decomposition with rounding.
    
    # Given the complexity, let's implement a simple search for rank 23 (for 3x3x3) and extend.
    # Actually, for 3x3x7, a simple approach is to use the 3x3x3 decomposition for each 3x3 block.
    # There are 7 such blocks (since 7 = 2*3 + 1), so rank would be 2*23 + 23 = 69? No, that's worse.
    # Better: Use the fact that 7 = 3+3+1.
    
    # Let's try a different approach: use a known 3x3x3 decomposition of rank 23 and combine.
    # But implementing a specific rank-23 decomposition is complex.
    
    # Alternative: Use a randomized search with small integer entries.
    # We'll set up a constraint satisfaction problem.
    
    # For simplicity and reliability within time limits, let's stick with schoolbook
    # unless we can quickly find a better decomposition.
    
    # Actually, let's try a simple Strassen-like reduction for the 3x3 case.
    # For 3x3x3, rank 23 is known but complex.
    
    # Let's use a more practical approach: try to find a decomposition with rank 40-50
    # using a greedy algorithm.
    
    # Due to time constraints, I'll implement a basic search that tries to reduce rank
    # by finding linear dependencies.
    
    # For now, return schoolbook as a safe fallback.
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
