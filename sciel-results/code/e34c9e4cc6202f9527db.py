"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val - round(val)) > tol and abs(val - (round(2*val)/2.0)) > tol:
                    return False
    return True

def _compute_residual_fast(U, V, W, n, m, p, target_rank):
    """Compute max residual efficiently."""
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(target_rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def _search_rank(n, m, p, target_rank, num_trials=50, max_iters=100, seed=42, time_limit=10.0):
    """
    Search for a rank-`target_rank` decomposition using randomized local search.
    Returns (U, V, W) as lists or None.
    """
    start = time.time()
    rng = random.Random(seed)
    
    # Allowed values: half-integers in a reasonable range
    allowed_vals = [v/2.0 for v in range(-4, 5)]
    
    best_U = None
    best_V = None
    best_W = None
    best_res = float('inf')
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        U = np.array([[rng.choice(allowed_vals) for _ in range(target_rank)] for _ in range(n*m)])
        V = np.array([[rng.choice(allowed_vals) for _ in range(target_rank)] for _ in range(m*p)])
        W = np.array([[rng.choice(allowed_vals) for _ in range(target_rank)] for _ in range(p*n)])
        
        res = _compute_residual_fast(U, V, W, n, m, p, target_rank)
        if res < best_res:
            best_res = res
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if res < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            mat_idx = rng.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                nrows = n*m
            elif mat_idx == 1:
                mat = V
                nrows = m*p
            else:
                mat = W
                nrows = p*n
            
            row = rng.randint(0, nrows-1)
            col = rng.randint(0, target_rank-1)
            old_val = mat[row, col]
            new_val = rng.choice(allowed_vals)
            if new_val == old_val:
                continue
            mat[row, col] = new_val
            
            res = _compute_residual_fast(U, V, W, n, m, p, target_rank)
            if res >= best_res:
                mat[row, col] = old_val
            else:
                best_res = res
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if res < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
    
    if best_U is not None and best_res < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    remaining = deadline - time.time()
    num_ranks_to_try = 8
    time_per_rank = remaining / num_ranks_to_try
    
    for target_rank in range(44, 44 - num_ranks_to_try, -1):
        if time.time() > deadline:
            break
        
        time_alloc = min(time_per_rank, deadline - time.time())
        if time_alloc < 1.0:
            break
        
        result = _search_rank(n, m, p, target_rank, num_trials=30, max_iters=50, 
                              seed=target_rank*7919, time_limit=time_alloc)
        
        if result is not None:
            U_list, V_list, W_list = result
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-9):
                if _check_half_integer(U_list, V_list, W_list, tol=1e-9):
                    best_rank = target_rank
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
