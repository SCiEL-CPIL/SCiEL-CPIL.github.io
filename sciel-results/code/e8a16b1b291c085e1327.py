import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n=2, m=3, p=5, tol=1e-6):
    """Verify that U, V, W form a valid tensor decomposition."""
    R = U.shape[1]
    # Check: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == delta_{j,j'}
    # for all i, j, j', k
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + i, r]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _schoolbook(n=2, m=3, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(R, n=2, m=3, p=5, time_limit=10):
    """Try to find a decomposition of rank R using random restarts and local search."""
    start_time = time.time()
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize randomly
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Normalize to avoid trivial scaling
        for r in range(R):
            norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
            if norm > 0:
                scale = norm ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # Local search: try to reduce the error
        error = _compute_error(U, V, W, n, m, p)
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            if error < 1e-8:
                return U, V, W, True
            
            # Pick a random column and try to improve it
            r = np.random.randint(R)
            # Compute gradient-like update for column r
            # This is simplified - just do a random perturbation
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            
            U_new[:, r] += 0.1 * np.random.randn(n * m)
            V_new[:, r] += 0.1 * np.random.randn(m * p)
            W_new[:, r] += 0.1 * np.random.randn(p * n)
            
            new_error = _compute_error(U_new, V_new, W_new, n, m, p)
            if new_error < error:
                U, V, W = U_new, V_new, W_new
                error = new_error
    
    return None, None, None, False

def _compute_error(U, V, W, n=2, m=3, p=5):
    """Compute the sum of squared errors of the decomposition."""
    R = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + i, r]
                    expected = 1.0 if j == jp else 0.0
                    error += (s - expected) ** 2
    return error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 30
    
    # Try to find lower rank solutions
    for R in range(25, best_rank):
        if time.time() - start_time > 100:
            break
        U, V, W, success = _try_rank(R, n, m, p, time_limit=10)
        if success:
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-5):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break  # Found a valid lower rank, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
