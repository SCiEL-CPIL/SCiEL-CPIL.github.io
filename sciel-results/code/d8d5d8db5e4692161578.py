"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _compute_error(U, V, W, n, m, p, rank):
    """Compute sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                err += (s - 1.0) ** 2
    return err

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _local_search(U, V, W, n, m, p, rank, max_iters=1000, rng=None):
    """Local search: flip individual entries to reduce error."""
    if rng is None:
        rng = random.Random()
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    err = _compute_error(U, V, W, n, m, p, rank)
    best_err = err
    
    for it in range(max_iters):
        if err < 1e-9:
            return U, V, W, err
        
        # Pick random entry
        which = rng.randint(0, 2)
        if which == 0:
            idx = rng.randint(0, n*m - 1)
            col = rng.randint(0, rank - 1)
            old = U[idx][col]
            U[idx][col] = rng.choice(vals)
        elif which == 1:
            idx = rng.randint(0, m*p - 1)
            col = rng.randint(0, rank - 1)
            old = V[idx][col]
            V[idx][col] = rng.choice(vals)
        else:
            idx = rng.randint(0, p*n - 1)
            col = rng.randint(0, rank - 1)
            old = W[idx][col]
            W[idx][col] = rng.choice(vals)
        
        new_err = _compute_error(U, V, W, n, m, p, rank)
        
        if new_err >= err:
            if which == 0:
                U[idx][col] = old
            elif which == 1:
                V[idx][col] = old
            else:
                W[idx][col] = old
        else:
            err = new_err
            if err < best_err:
                best_err = err
    
    return U, V, W, err

def _try_rank(n, m, p, rank, num_trials=30, seed=42, max_iters=500):
    """Try to find a rank-r decomposition."""
    rng = random.Random(seed)
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(num_trials):
        U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
        
        U, V, W, err = _local_search(U, V, W, n, m, p, rank, max_iters=max_iters, rng=rng)
        
        if err < best_err:
            best_err = err
            best_result = (U, V, W)
        
        if err < 1e-9:
            break
    
    if best_result is not None and best_err < 1e-9:
        return best_result
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, num_trials=20, seed=rank*7+1, max_iters=300)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END

