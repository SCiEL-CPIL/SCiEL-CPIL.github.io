import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find lower rank
    # Known lower bound for 2x5x5 is 21
    for target_rank in range(21, rank_schoolbook):
        if time.time() > deadline:
            break
        success = _try_rank(n, m, p, target_rank)
        if success:
            best_rank = target_rank
            # Extract best_U, best_V, best_W from _try_rank or re-run
            # _try_rank returns None, so we need to store the solution
            # Let's modify _try_rank to return the solution
            sol = _try_rank_solve(n, m, p, target_rank)
            if sol:
                best_U, best_V, best_W = sol
                break
        
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_rank_solve(n, m, p, rank):
    # This function tries to find a decomposition of the given rank
    # Returns (U, V, W) or None
    pass

def _try_rank(n, m, p, rank):
    # Placeholder
    return False
