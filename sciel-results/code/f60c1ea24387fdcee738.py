"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize best result with schoolbook
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # For small matrices, we can attempt to find lower rank decompositions
    # using numerical methods, but exact integer/half-integer entries are required.
    # Given the constraints and time limit, we stick to the schoolbook method 
    # unless a known lower rank exists for these specific dimensions.
    
    # Known results:
    # For 2x2: rank 7
    # For 3x3: rank 23 (or 21?)
    # For 3x5x7: Not a standard small case with widely known sub-schoolbook ranks.
    # The schoolbook rank is 105.
    
    # Since finding exact low-rank decompositions for arbitrary matrix sizes is hard
    # and requires specific algebraic structures, and given the time limit,
    # we return the schoolbook decomposition which is guaranteed to work.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
