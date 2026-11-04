import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Fallback: Schoolbook
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook

    # Try to find rank 19 decomposition
    # Based on BCLR 5x5x5 rank 19, restricted to 4x5x5
    # We will attempt to solve for U, V, W with 19 columns
    
    target_rank = 19
    
    # We need U (20x19), V (25x19), W (20x19)
    # such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # This is a system of 100 equations.
    # We can use a random initialization and gradient descent or least squares.
    # Since coefficients are expected to be small integers, we can try to find a sparse solution.
    
    # Strategy: Randomly initialize U, V, W with small integers {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    # and check if it works. If not, use a local search.
    
    # Given the complexity, we'll try a few random seeds.
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(seed)
        
        # Initialize with small integers
        U = np.random.randint(-1, 2, (n * m, target_rank))
        V = np.random.randint(-1, 2, (m * p, target_rank))
        W = np.random.randint(-1, 2, (p * n, target_rank))
        
        # Verify
        if _verify(n, m, p, U, V, W):
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break

    # If random search failed, try a structured approach based on BCLR if available
    # For now, if 19 not found, return best found (schoolbook)
    
    return best_result

def _verify(n, m, p, U, V, W):
    rank = U.shape[1]
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True
