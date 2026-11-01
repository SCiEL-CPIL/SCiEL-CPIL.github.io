"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition
    # Strategy: Use a randomized approach with iterative refinement
    for target_rank in range(45, rank_school):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        np.random.seed(target_rank)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Try to optimize to satisfy the constraints
        # The constraint is: sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c] = delta_{j,j'}*delta_{k,k'}*delta_{i,i'}
        
        # This is a hard non-linear optimization problem.
        # For now, just check if a simple structure works.
        
        # Try a structured decomposition
        # For example, use the fact that we can factor through smaller matrices
        
        if target_rank == 49:
            # Try to merge two schoolbook terms
            U_test = U_school.copy()
            V_test = V_school.copy()
            W_test = W_school.copy()
            
            # Remove the last column and see if we can adjust
            # This is a heuristic; may not work
            pass
        
        # Check validity of a simple candidate
        # For rank 1: all ones
        if target_rank == 1:
            U_cand = np.ones((n * m, 1))
            V_cand = np.ones((m * p, 1))
            W_cand = np.ones((p * n, 1))
            # Check: sum_c U[i*m+j,c] V[j'*p+k,c] W[k'*n+i',c]
            # = U[i*m+j,0] * V[j'*p+k,0] * W[k'*n+i',0] = 1
            # But we need delta_{j,j'}*delta_{k,k'}*delta_{i,i'}
            # So this doesn't work.
            pass
    
    # Return the best result found (schoolbook for now)
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
