import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_exact(U, V, W, n, m, p, r, tol=1e-9):
    """Verify that the decomposition is exact (within tolerance)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, r

def _residual_norm(U, V, W, n, m, p, r):
    """Compute the L2 norm of the residual tensor."""
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                total += (s - 1.0) ** 2
    return np.sqrt(total)

def _snap_to_half_int(arr, tol=1e-5):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2.0
    if np.allclose(arr, snapped, atol=tol):
        return snapped, True
    return arr, False

def _als_decompose(n, m, p, r, max_iter=300, tol=1e-12, time_budget=5.0, seed=0):
    """
    Alternating Least Squares to find rank-r decomposition.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with multiple strategies
    strategy = seed % 3
    if strategy == 0:
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
    elif strategy == 1:
        # Sparse initialization
        U = rng.choice([-1, 0, 1], size=(n*m, r)) * 0.5
        V = rng.choice([-1, 0, 1], size=(m*p, r)) * 0.5
        W = rng.choice([-1, 0, 1], size=(p*n, r)) * 0.5
    else:
        # Block-structured initialization
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        for l in range(r):
            i = l % n
            j = (l // n) % m
            k = (l // (n*m)) % p
            U[i*m+j, l] = 1.0
            V[j*p+k, l] = 1.0
            W[k*n+i, l] = 1.0
        # Add random perturbation
        U += rng.randn(n*m, r) * 0.1
        V += rng.randn(m*p, r) * 0.1
        W += rng.randn(p*n, r) * 0.1
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for l in range(r):
                        A[k, l] = V[j*p+k, l] * W[k*n+i, l]
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for l in range(r):
                        A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    for l in range(r):
                        A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass
        
        # Compute error
        error = _residual_norm(U, V, W, n, m, p, r)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return U, V, W, error

def _polish(U, V, W, n, m, p, r, max_iter=100, time_budget=3.0):
    """High-precision polish pass."""
    start_time = time.time()
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for l in range(r):
                        A[k, l] = V[j*p+k, l] * W[k*n+i, l]
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-15)
                    U[i*m+j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for l in range(r):
                        A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-15)
                    V[j*p+k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    for l in range(r):
                        A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-15)
                    W[k*n+i, :] = w
                except:
                    pass
    
    error = _residual_norm(U, V, W, n, m, p, r)
    return U, V, W, error

def _try_rank(n, m, p, r, max_seeds=80, time_per_seed=1.5):
    """Try to find a valid rank-r decomposition."""
    start_time = time.time()
    deadline = start_time + 120
    
    for seed in range(max_seeds):
        if time.time() > deadline:
            break
        
        # Run ALS
        U, V, W, error = _als_decompose(n, m, p, r, max_iter=300, tol=1e-12, time_budget=time_per_seed, seed=seed)
        
        if error < 1e-4:
            # Polish
            U, V, W, error = _polish(U, V, W, n, m, p, r, max_iter=100, time_budget=1.0)
        
        if error < 1e-5:
            # Snap to half-integers
            U_snap, ok1 = _snap_to_half_int(U)
            V_snap, ok2 = _snap_to_half_int(V)
            W_snap, ok3 = _snap_to_half_int(W)
            
            if ok1 and ok2 and ok3:
                if _verify_exact(U_snap, V_snap, W_snap, n, m, p, r, tol=1e-8):
                    return U_snap, V_snap, W_snap, r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_result = {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
    best_rank = rank_sb
    
    # Try ranks from 20 upward
    for r in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, r, max_seeds=100, time_per_seed=1.2)
        
        if result is not None:
            U_snap, V_snap, W_snap, rank = result
            best_result = {
                "U": [[float(x) for x in row] for row in U_snap],
                "V": [[float(x) for x in row] for row in V_snap],
                "W": [[float(x) for x in row] for row in W_snap]
            }
            best_rank = rank
            break
    
    return best_result
