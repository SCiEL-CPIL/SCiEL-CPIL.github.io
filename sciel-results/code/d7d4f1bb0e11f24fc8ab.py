"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_T0 = time.time()
_DEADLINE = _T0 + 110  # safety margin before 120s hard limit

def _elapsed():
    return time.time() - _T0

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_improve(n, m, p, max_rank=72, time_budget=100):
    """Try to find a lower-rank decomposition."""
    best = None
    best_rank = n * m * p
    
    # Try ranks from best_rank-1 down to some minimum
    # For each rank, try a random search
    
    for rank in range(best_rank - 1, max(20, best_rank - 30), -1):
        if _elapsed() > time_budget:
            break
        
        # Try a few random attempts for this rank
        attempts = 5
        for attempt in range(attempts):
            if _elapsed() > time_budget:
                break
            
            # Random initialization with small integer/half-integer values
            np.random.seed(hash((rank, attempt)) % (2**32))
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, rank))
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = (U, V, W)
                best_rank = rank
                break
        
        if best is not None:
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    
    # Try to improve
    best, best_rank = _try_improve(n, m, p, max_rank=72, time_budget=100)
    
    if best is not None:
        U, V, W = best
    else:
        U, V, W = U_sb, V_sb, W_sb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
