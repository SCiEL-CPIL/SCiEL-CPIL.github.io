import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline to stay well under 150s

    # 1. Construct Schoolbook Decomposition (Rank n*m*p = 48)
    rank_sb = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # 2. Search for lower rank decompositions
    # Strategy: Try ranks from low to high, but prioritize likely minimal ranks
    # For 3x4x4, the minimal rank is likely between 20-30
    # Try ranks in a strategic order:
    # First try 20-25 (likely minimal range) with more attempts
    # Then 26-35 with fewer attempts
    # Then 36-47 if time permits
    
    # Actually, let's be smarter:
    # The schoolbook rank is 48.
    # We know that for small matrix multiplication, the rank can be significantly lower.
    # For 3x4x4, let's try a range of ranks.
    
    # Key insight: We should try to find ANY valid decomposition with rank < 48
    # Start from ranks that are likely achievable and work our way up
    # But we want the MINIMAL rank, so we should try low ranks first
    
    # However, low ranks are harder to find. 
    # Let's try a balanced approach:
    # Try ranks 25-35 first (more likely to find something), then 20-24, then 36-47
    
    # Actually, let's just try from low to high, but with time-limited attempts
    # If we find a valid rank R, we record it and try to find R-1, etc.
    # But that's expensive. Instead, let's just find the lowest rank we can.
    
    # Revised strategy:
    # Try ranks in order: 20, 21, 22, ..., 47
    # For each rank, spend a limited amount of time trying to find a decomposition
    # If found, record it and continue trying lower ranks (if going down) or stop (if going up)
    
    # Since we want minimal, let's go from low to high:
    # If we find rank 25, we don't know if 20 is possible.
    # So we should continue trying 20-24.
    # But that's expensive.
    
    # Better: Try a few promising ranks first, then expand.
    # Let's try: 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
    # Then 20, 21, 22, 23, 24 if time remains
    
    # Actually, for simplicity and to maximize chance of finding something:
    # Try ranks from 47 down to 20, keep the best (lowest) found
    # This way, we're more likely to find higher ranks first (easier), 
    # then try to find lower ones
    
    search_order = list(range(47, 19, -1))  # 47, 46, ..., 20
    
    for rank in search_order:
        if time.time() > deadline - 5:
            break
        
        # Time per rank: give more time to lower ranks (harder)
        if rank < 25:
            time_limit = 3.0
            max_restarts = 5
        elif rank < 35:
            time_limit = 2.0
            max_restarts = 3
        else:
            time_limit = 1.5
            max_restarts = 2
        
        found = _try_find_rank(n, m, p, rank, 
                               max_restarts=max_restarts, 
                               time_limit=time_limit, 
                               deadline=deadline)
        
        if found is not None:
            U, V, W = found
            if rank < best_rank:
                best_rank = rank
                best_result = found
                # If we find a very low rank, we can stop
                if rank <= 22:
                    break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
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

def _try_find_rank(n, m, p, rank, max_restarts=5, time_limit=3, deadline=None):
    start_time = time.time()

    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        if deadline is not None and time.time() > deadline - 3:
            break

        # Initialize
        if restart % 2 == 0:
            U, V, W = _init_prune(n, m, p, rank)
        else:
            U, V, W = _init_random(n, m, p, rank)

        # ALS iterations
        U, V, W, max_err = _als(U, V, W, n, m, p,
                                 iters=30,
                                 start_time=start_time,
                                 time_limit=time_limit)

        # Check if close enough
        if max_err < 1e-5:
            # Try to polish and snap
            snapped, ok = _snap_and_check(U, V, W, n, m, p)
            if ok:
                return snapped

    return None

def _init_prune(n, m, p, rank):
    """Start from schoolbook, keep 'rank' random terms, add perturbation."""
    nmp = n * m * p
    U, V, W = _schoolbook(n, m, p)

    # Choose 'rank' random columns to keep
    cols = random.sample(range(nmp), rank)
    U = U[:, cols]
    V = V[:, cols]
    W = W[:, cols]

    # Add small perturbation
    U += np.random.randn(n * m, rank) * 0.1
    V += np.random.randn(m * p, rank) * 0.1
    W += np.random.randn(p * n, rank) * 0.1
    return U, V, W

def _init_random(n, m, p, rank):
    """Random initialization with structure."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(rank):
        # For each rank component, create a rank-1 tensor
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # The product sum should give 1 for (i,j,k)
        
        # Initialize with small random values
        U[:, r] = np.random.randn(n * m) * 0.5
        V[:, r] = np.random.randn(m * p) * 0.5
        W[:, r] = np.random.randn(p * n) * 0.5

    return U, V, W

def _als(U, V, W, n, m, p, iters=30, start_time=None, time_limit=5):
    """Alternating least squares for tensor decomposition."""
    max_err = float('inf')
    prev_err = float('inf')

    for it in range(iters):
        if start_time is not None and time.time() > start_time + time_limit:
            break

        # Update U: for each (i,j), solve V_part * W_part * U_vec = 1
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # V_part: rows j*p to (j+1)*p, all columns
                V_part = V[j * p:(j + 1) * p, :]  # shape (p, rank)
                # W_part: rows k*n+i for k=0..p-1
                W_indices = [k * n + i for k in range(p)]
                W_part = W[W_indices, :]  # shape (p, rank)
                
                # A = V_part * W_part (element-wise), shape (p, rank)
                A = V_part * W_part
                b = np.ones(p)
                
                try:
                    # Solve A * x = b for x = U[idx, :]
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except Exception:
                    U_new[idx, :] = U[idx, :]
        U = U_new

        # Update V: for each (j,k), solve U_part * W_part * V_vec = 1
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                # U_part: rows i*m+j for i=0..n-1
                U_indices = [i * m + j for i in range(n)]
                U_part = U[U_indices, :]  # shape (n, rank)
                # W_part: rows k*n to (k+1)*n
                W_part = W[k * n:(k + 1) * n, :]  # shape (n, rank)
                
                A = U_part * W_part
                b = np.ones(n)
                
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except Exception:
                    V_new[idx, :] = V[idx, :]
        V = V_new

        # Update W: for each (k,i), solve U_part * V_part * W_vec = 1
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                # U_part: rows i*m to (i+1)*m
                U_part = U[i * m:(i + 1) * m, :]  # shape (m, rank)
                # V_part: rows j*p+k for j=0..m-1
                V_indices = [j * p + k for j in range(m)]
                V_part = V[V_indices, :]  # shape (m, rank)
                
                A = U_part * V_part
                b = np.ones(m)
                
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except Exception:
                    W_new[idx, :] = W[idx, :]
        W = W_new

        # Compute max error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    e = abs(val - 1.0)
                    if e > max_err:
                        max_err = e

        if max_err < 1e-8:
            break
        if abs(prev_err - max_err) < 1e-12:
            break
        prev_err = max_err

    return U, V, W, max_err

def _snap_and_check(U, V, W, n, m, p, tol=1e-4):
    """Snap to half-integers and check exact decomposition."""
    # Try different snapping strategies
    for scale in [2, 4, 1]:
        U_s = np.round(U * scale) / scale
        V_s = np.round(V * scale) / scale
        W_s = np.round(W * scale) / scale
        
        # Check if close to original
        if (np.max(np.abs(U - U_s)) < tol and
            np.max(np.abs(V - V_s)) < tol and
            np.max(np.abs(W - W_s)) < tol):
            
            # Check exact decomposition
            if _check_exact(U_s, V_s, W_s, n, m, p):
                return (U_s, V_s, W_s), True
    
    return None, False

def _check_exact(U, V, W, n, m, p, tol=1e-10):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
