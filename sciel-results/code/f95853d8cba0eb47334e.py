"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
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

def _random_attempt(n, m, p, target_rank, seed=None, candidates=None):
    """Generate a random decomposition and verify it."""
    if seed is not None:
        random.seed(seed)
    
    if candidates is None:
        candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    U_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(n*m)]
    V_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(m*p)]
    W_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(p*n)]
    
    if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r, V_r, W_r
    return None, None, None

def _try_rank(n, m, p, target_rank, num_attempts=100, start_seed=0, time_budget=5.0):
    """Try to find a decomposition of given rank with multiple random attempts."""
    start_time = time.time()
    
    for attempt in range(num_attempts):
        if time.time() - start_time > time_budget:
            break
        
        seed = start_seed + attempt * 137
        U_r, V_r, W_r = _random_attempt(n, m, p, target_rank, seed=seed)
        
        if U_r is not None:
            return U_r, V_r, W_r
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave margin for 150s limit
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Known achievable rank for 3x5x5 is around 40
    # Try ranks from 40 down to lower bound
    
    # Use a wider set of candidates for better coverage
    candidates_wide = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    
    # First pass: try ranks from 40 down, with quick attempts
    for target_rank in range(40, 24, -1):
        if time.time() > deadline - 10:
            break
        
        # Quick attempt with fewer restarts
        U_r, V_r, W_r = _try_rank(
            n, m, p, target_rank, 
            num_attempts=50, 
            start_seed=42,
            time_budget=3.0
        )
        
        if U_r is not None:
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    # Second pass: if we found something, try to improve with more attempts
    if best_rank < n * m * p:
        for target_rank in range(40, 24, -1):
            if time.time() > deadline - 5:
                break
            if target_rank >= best_rank:
                continue
            
            U_r, V_r, W_r = _try_rank(
                n, m, p, target_rank, 
                num_attempts=100, 
                start_seed=1000,
                time_budget=5.0
            )
            
            if U_r is not None:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    return best_result
# EVOLVE-BLOCK-END
