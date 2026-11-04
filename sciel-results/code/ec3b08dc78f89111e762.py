"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Schoolbook decomposition as baseline (rank 42)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Strategy: Algebraic/Coding Theory inspired search.
    # We look for a rank-r decomposition where r < 42.
    # The tensor is T[i,j,k] = 1.
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    # This is a system of 42 equations.
    # Variables: U (nm*r), V (mp*r), W (pn*r).
    # We restrict entries to small integers/half-integers to facilitate exactness.
    
    # Known lower bound for 2x3x7 is 21 (from max(nm, mp, pn) / 2? No, max is 21).
    # Actually, max(nm, mp, pn) = 21.
    # So rank >= 21.
    # We try ranks from 21 upwards.
    
    # To improve the search, we use a "structured" random initialization.
    # Instead of pure random, we try to align with the Kronecker product structure 
    # or use a "parity-check" like constraint satisfaction approach.
    
    # For each target rank, we try to solve the system using a greedy/local search 
    # or randomized restarts with small integer values.
    
    min_rank = 21
    max_search_rank = best_rank - 1  # Try to improve
    
    # We will attempt to find a solution for each rank in [min_rank, max_search_rank]
    # We prioritize lower ranks.
    
    for target_rank in range(min_rank, max_search_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts per rank
        # For lower ranks, more attempts might be needed, but probability is low.
        # For ranks close to 42, it's easier.
        # Let's allocate attempts based on rank distance from 42.
        attempts = 200 if target_rank < 30 else 500
        
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random matrices with small integer entries
            # Using -1, 0, 1 is a good start for algebraic structures.
            # Sometimes half-integers (multiples of 0.5) are needed, but let's stick to integers first
            # as they are easier to verify exactly.
            
            # To incorporate the "coding theory" idea:
            # We can view the columns of U, V, W as vectors.
            # The condition is that the "trilinear form" evaluates to 1.
            
            # Random initialization with small integers
            # We use a biased distribution to encourage sparsity or specific patterns
            # which might correspond to code structures.
            
            def gen_matrix(rows, cols):
                mat = [[0] * cols for _ in range(rows)]
                for r in range(rows):
                    for c in range(cols):
                        # Mostly 0, sometimes -1 or 1
                        # This mimics a sparse code
                        if random.random() < 0.3:
                            mat[r][c] = random.choice([-1, 1])
                        # Occasionally 2?
                        elif random.random() < 0.1:
                            mat[r][c] = random.choice([-2, 2])
                return mat

            U = gen_matrix(nm, target_rank)
            V = gen_matrix(mp, target_rank)
            W = gen_matrix(pn, target_rank)
            
            if verify(U, V, W, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # If we found a solution, we can continue to try for even lower ranks
            # but we must ensure we don't exceed time.
            continue
        
        # If we didn't find a solution for this rank, we might still try the next one
        # but we should be careful with time.
        
    return best
# EVOLVE-BLOCK-END
