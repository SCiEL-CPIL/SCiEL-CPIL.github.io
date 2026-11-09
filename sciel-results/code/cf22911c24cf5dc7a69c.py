"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_random_search(n, m, p, target_rank, time_limit=10.0, num_trials=500):
    """Try random sparse search for a decomposition of given rank."""
    start = time.time()
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for c in range(target_rank):
            # Each rank component: pick sparse support with ±1 values
            u_support = random.sample(range(n*m), random.randint(1, 4))
            v_support = random.sample(range(m*p), random.randint(1, 4))
            w_support = random.sample(range(p*n), random.randint(1, 4))
            
            for idx in u_support:
                U[idx, c] = random.choice([-1, 1])
            for idx in v_support:
                V[idx, c] = random.choice([-1, 1])
            for idx in w_support:
                W[idx, c] = random.choice([-1, 1])
        
        if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify_decomp(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, target_rank):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _construct_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    deadline = time.time() + 110.0
    
    # Try to find better decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = 24
    # Try ranks from a reasonable minimum upward
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try structured ranks
    for target_rank in range(30, min(schoolbook_rank, 50) + 1):
        if time.time() > deadline - 20:
            break
        
        result = _try_random_search(n, m, p, target_rank, 
                                    time_limit=min(5.0, max(0.5, deadline - time.time() - 20)),
                                    num_trials=200)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break  # Found a valid one, stop
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
