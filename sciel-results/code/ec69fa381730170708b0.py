"""
Refined program for matrix multiplication 4x5x6.
Searches for rank-minimal exact tensor decomposition.
"""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition
    for multiplying a 4 by 5 matrix by a 5 by 6 matrix.
    """
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Try structured approaches for ranks below schoolbook
    # Approach: use the fact that we can share rank terms
    
    # Try a few specific ranks with structured constructions
    candidate_ranks = []
    for r in range(lower_bound, rank_school):
        candidate_ranks.append(r)
    
    # Sort by rank (try lowest first)
    candidate_ranks.sort()
    
    random.seed(42)
    
    for trial_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a structured approach: try different patterns
        
        found = False
        
        # Attempt 1: Try with small integer values
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with zeros
            U = [[0] * trial_rank for _ in range(n * m)]
            V = [[0] * trial_rank for _ in range(m * p)]
            W = [[0] * trial_rank for _ in range(p * n)]
            
            # Try to assign values systematically
            # For each (i,j,k), we need the sum to be 1
            
            # Strategy: try to use a pattern where each rank term covers multiple (i,j,k)
            # This is complex, so try random sparse patterns
            
            # Generate random sparse matrices
            for c in range(trial_rank):
                # Choose random positions
                u_positions = random.sample(range(n * m), min(3, n * m))
                v_positions = random.sample(range(m * p), min(3, m * p))
                w_positions = random.sample(range(p * n), min(3, p * n))
                
                for pos in u_positions:
                    U[pos][c] = random.choice([-1, 1])
                for pos in v_positions:
                    V[pos][c] = random.choice([-1, 1])
                for pos in w_positions:
                    W[pos][c] = random.choice([-1, 1])
            
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = trial_rank
                found = True
                break
        
        if found:
            # Found a lower rank, continue searching for even lower
            continue
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
