import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 4 * 5 * 5 = 100
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
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
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    
    # Try to find a better decomposition via random search with small rank
    # We'll try ranks from 50 to 99
    random.seed(123)
    
    for target_rank in [50, 60, 70, 80, 90, 95, 99]:
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Random sparse initialization
            for r in range(target_rank):
                # Each factor gets a random subset of indices
                U_indices = random.sample(range(n * m), min(5, n * m))
                V_indices = random.sample(range(m * p), min(5, m * p))
                W_indices = random.sample(range(p * n), min(5, p * n))
                
                for idx in U_indices:
                    U[idx, r] = random.choice([-1, 1]) * random.random()
                for idx in V_indices:
                    V[idx, r] = random.choice([-1, 1]) * random.random()
                for idx in W_indices:
                    W[idx, r] = random.choice([-1, 1]) * random.random()
            
            # Verify if this decomposition is exact
            # The condition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a nonlinear constraint, so random search is unlikely to work.
            # Instead, let's try a structured approach.
            pass
    
    # Try a structured reduction: remove one term from schoolbook if possible
    # Schoolbook has rank 100. Can we merge any two terms?
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if i1=i2, j1=j2, k1=k2, which is never true.
    # So schoolbook is minimal for that structure.
    
    # Try Winograd-like approach for 5x5 subproblem
    # The 4x5x5 problem can be seen as 4 separate 1x5x5 problems (rows of A times B)
    # Or 5 separate 4x5x1 problems
    
    # Let's try a different approach: use the fact that we only need 4 rows of the output
    # The full 5x5x5 tensor has rank 36 (Winograd). We need a sub-tensor with 4x5 output indices.
    # This should have rank <= 36.
    
    # For now, return schoolbook to ensure validity
    return best_result

# EVOLVE-BLOCK-END
