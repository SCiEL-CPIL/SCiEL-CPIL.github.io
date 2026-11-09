"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s to leave margin
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank via randomized search
    # Use small integer/half-integer values
    candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for target_rank in range(40, best_rank):  # Try ranks from 40 up to current best
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(n*m)]
            V_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(m*p)]
            W_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break  # Found valid decomposition for this rank, try lower
        
        if best_rank == target_rank:
            # Found a decomposition at this rank, continue to try lower
            continue
        else:
            # If we couldn't find at this rank, and we're going upward, stop
            # Actually we're iterating from low to high, so if we fail at target_rank, 
            # we might still find at higher ranks. But we want the lowest.
            # Let's continue trying higher ranks if we didn't find at this one
            pass
    
    return best_result
# EVOLVE-BLOCK-END
