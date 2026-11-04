"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(s - 1.0) > tol:
                    return False
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                s = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(s) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
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

def _try_rank_reduced(n, m, p, target_rank, time_limit=30.0):
    """Try to find a decomposition with target_rank using random search with structure."""
    if n != 2:
        return None
    
    start = time.time()
    deadline = start + time_limit
    best = None
    best_err = float('inf')
    
    # Use structured random initialization
    # U: (10, r), V: (25, r), W: (10, r)
    # We'll use a parametric family to reduce search space
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        r = target_rank
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Compute error
        U_t = U.T  # (r, 10)
        V_t = V.T  # (r, 25)
        W_t = W.T  # (r, 10)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U_t[:, a] * V_t[:, b] * W_t[:, c])
                    err += (s - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            break
    
    if best is None or best_err > 1e-6:
        return None
    
    U_best, V_best, W_best = best
    
    # Convert to list of lists with rounded values
    U_out = [[round(x, 6) for x in row] for row in U_best.tolist()]
    V_out = [[round(x, 6) for x in row] for row in V_best.tolist()]
    W_out = [[round(x, 6) for x in row] for row in W_best.tolist()]
    
    # Verify
    if _verify(U_out, V_out, W_out, n, m, p):
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with schoolbook (rank 50)
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to find lower rank decompositions
    # Try ranks from 49 down to 40, spending limited time on each
    for target_rank in range(best_rank - 1, max(30, best_rank - 15), -1):
        if not still_going():
            break
        
        remaining_time = deadline - time.time()
        time_for_attempt = min(20.0, remaining_time * 0.5)
        
        result = _try_rank_reduced(n, m, p, target_rank, time_limit=time_for_attempt)
        if result is not None:
            best = result
            best_rank = target_rank
            break  # Found a valid lower rank, stop
    
    return best
# EVOLVE-BLOCK-END
