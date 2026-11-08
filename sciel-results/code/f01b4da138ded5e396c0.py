"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Default: Schoolbook decomposition
    # Rank = n * m * p
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition using a simple search
    # We'll try ranks from a lower bound up to schoolbook rank
    # Lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n)
    # For 3x5x7: max(15, 35, 21) = 35
    
    # Try a few candidate ranks below schoolbook
    # Use a randomized approach to search for valid decompositions
    
    # First, let's try if there's a known decomposition pattern
    # For now, try a simple local search starting from schoolbook
    
    # Attempt to reduce rank by checking for linear dependencies
    # This is a heuristic approach
    
    # Given time constraints, we'll do a limited search
    # Try to find a decomposition with rank < 105
    
    # Simple approach: try to merge some terms
    # For each pair of terms, check if they can be combined
    
    # Actually, let's try a different approach: use a known result
    # The minimum rank for 3x5x7 matrix multiplication is not well-known,
    # but we can try to find an improvement using algebraic methods
    
    # For now, return the schoolbook as it's guaranteed valid
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
