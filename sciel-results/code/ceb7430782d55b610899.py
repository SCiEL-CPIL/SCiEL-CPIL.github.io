"""Matrix multiplication 3x4x4: find rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def _validate(U, V, W):
    """Check if U, V, W exactly decompose the 3x4x4 matmul tensor."""
    r = len(U[0])
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank(r, time_limit=5, seed=0):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Reduced attempts to prevent timeout
    start = time.time()
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small values
        U = np_rng.randn(n*m, r) * 0.5
        V = np_rng.randn(m*p, r) * 0.5
        W = np_rng.randn(p*n, r) * 0.5
        
        # Iterative refinement: update one factor at a time
        for iteration in range(50):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            
            a = i * m + j
            b = j * p + k
            c_idx = k * n + i
            
            # Current value
            cur = np.dot(U[a], V[b] * W[c_idx])
            err = 1.0 - cur
            
            lr = 0.1
            U[a] += lr * err * V[b] * W[c_idx]
            V[b] += lr * err * U[a] * W[c_idx]
            W[c_idx] += lr * err * U[a] * V[b]
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a], V[b] * W[c_idx])
                    err_sum += (val - 1)**2
        
        if err_sum < best_err:
            best_err = err_sum
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < 1e-12:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit
    
    # Try ranks from low to high
    for r in range(20, 49):
        if time.time() - start_time > deadline:
            break
        
        # Try a few seeds for this rank
        for seed in range(5):
            if time.time() - start_time > deadline:
                break
            
            result = _try_rank(r, time_limit=min(10, deadline - (time.time() - start_time)), seed=seed)
            if result is not None:
                U, V, W = result
                # Validate
                if _validate(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
