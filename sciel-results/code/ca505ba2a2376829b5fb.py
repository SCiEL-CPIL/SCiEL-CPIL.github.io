"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_candidate(r, n, m, p):
    """Generate a random candidate with entries in {-1, 0, 1}."""
    U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
    V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
    W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s
    
    # Schoolbook as default
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 144
    
    # Search for lower ranks
    # Based on known results, rank of M_{4,4,9} is likely well below 144
    # Try from a reasonable lower bound upward
    min_rank = 50  # conservative lower bound
    
    for r in range(min_rank, best_rank):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            U, V, W = _random_candidate(r, n, m, p)
            if _verify_fast(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
            attempts += 1
        
        if best_rank == r:
            break
    
    return best
# EVOLVE-BLOCK-END
