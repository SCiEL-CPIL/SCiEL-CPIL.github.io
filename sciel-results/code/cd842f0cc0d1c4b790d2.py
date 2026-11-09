"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition"""
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

def _try_rank_r(n, m, p, r, seed, max_iters=500):
    """Try to find a rank-r decomposition using random search"""
    random.seed(seed)
    np.random.seed(seed)
    
    for it in range(max_iters):
        # Random initialization with small values
        U = np.random.randint(-2, 3, size=(n*m, r)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, r)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, r)).astype(float)
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with trivial decomposition
    best = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n*m, m*p, p*n)
    
    seed = 42
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 30:
            break
        result = _try_rank_r(n, m, p, r, seed)
        if result is not None:
            best = result
            best_rank = r
            break
        seed += 1
    
    return best
# EVOLVE-BLOCK-END
