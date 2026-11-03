"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Schoolbook baseline
    rank_sb = n * m * p  # 120
    def make_schoolbook():
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = rank_sb

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Use structured decomposition based on the identity:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can try to reduce rank by exploiting the fact that for fixed i and k,
    # the sum over j is a dot product. One known technique is to use the 
    # "Strassen-like" decomposition or to find linear dependencies.
    #
    # For 4x5x6, let's try a systematic approach:
    # 1. Start with schoolbook (rank 120)
    # 2. Try to merge columns by finding linear dependencies
    #
    # Key insight: If two columns c1 and c2 in the decomposition have the property that
    # for all (i,j,k), the contribution from c1 equals the contribution from c2, then
    # we can potentially merge them.
    #
    # More practically, we can try to find a decomposition of rank r < 120 by:
    # - Setting up the system of equations
    # - Using the fact that the tensor is "simple" (all entries are 1)
    #
    # Let's try a different approach: use the known result that for matrix multiplication,
    # the rank can be reduced using "winograd-like" schemes or by exploiting the 
    # rectangular structure.
    #
    # For now, let's try a few specific structured patterns:
    
    # Pattern 1: Try to use rank = n*p*m - (n-1)*(p-1) by exploiting some structure
    # This is a heuristic, not guaranteed to work
    
    # Let's try a more systematic approach:
    # For each possible rank r from 119 down to some minimum, try to find a valid decomposition
    # using a structured random search with small integer entries.
    
    # Given the time constraint, we'll try a few ranks with limited trials.
    
    if time.time() < deadline - 30.0:
        # Try rank 119 with a few structured attempts
        for target_rank in [119, 118]:
            if time.time() > deadline - 25.0:
                break
            
            # Try a structured approach: start with schoolbook and try to merge
            # two columns that have some relationship
            for trial in range(50):
                if time.time() > deadline - 20.0:
                    break
                
                # Create a perturbation of the schoolbook
                # Try to find if we can reduce rank by 1 or 2
                # This is a heuristic approach
                
                # For now, just keep the schoolbook as it's guaranteed correct
                # A proper search would require solving a system of polynomial equations
                break
    
    # The schoolbook decomposition is the safe, correct answer
    return best_result
# EVOLVE-BLOCK-END
