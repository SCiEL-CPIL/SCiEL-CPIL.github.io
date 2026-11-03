"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _max_error(U, V, W, n, m, p):
    """Compute the maximum absolute error over all (i,j,k)."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _als_iterate(U, V, W, n, m, p, max_iter=300, tol=1e-14):
    """Alternating least squares for tensor decomposition."""
    rank = U.shape[1]
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U
        for idx in range(n * m):
            i, j = divmod(idx, m)
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        
        # Update V
        for idx in range(m * p):
            j, k = divmod(idx, p)
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V[idx, :] = sol
            except:
                pass
        
        # Update W
        for idx in range(p * n):
            k, i = divmod(idx, n)
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W[idx, :] = sol
            except:
                pass
        
        # Check convergence
        diff = (np.max(np.abs(U - U_old)) + 
                np.max(np.abs(V - V_old)) + 
                np.max(np.abs(W - W_old)))
        if diff < tol:
            break
    
    return U, V, W

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _try_snap_and_check(U, V, W, n, m, p):
    """Try to snap to half-integers and check exactness."""
    for tol in [1e-5, 1e-4, 1e-3]:
        U_snapped, u_ok = _snap_to_half_integer(U, tol=tol)
        V_snapped, v_ok = _snap_to_half_integer(V, tol=tol)
        W_snapped, w_ok = _snap_to_half_integer(W, tol=tol)
        if u_ok and v_ok and w_ok:
            if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped
    return None

def _structured_init_block_split(n, m, p, rank, seed=0):
    """
    Structured initialization: split the j-index into blocks and use
    block-structured rank-1 terms.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try to assign rank-1 terms to blocks of (i,j,k)
    # Each term covers a subset of (i,j,k) indices
    col = 0
    # First, use some terms for individual (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    # Fill remaining columns with random values
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.5
        V[:, col] = rng.randn(m*p) * 0.5
        W[:, col] = rng.randn(p*n) * 0.5
        col += 1
    
    return U, V, W

def _structured_init_strassen_embed(n, m, p, rank, seed=0):
    """
    Structured initialization inspired by Strassen: use sums/differences
    of entries to create shared rank-1 terms.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each (i,k), the sum over j is what we need.
    # Use terms that cover multiple j values.
    col = 0
    for i in range(n):
        for k in range(p):
            if col >= rank:
                break
            # Term that covers all j for this (i,k)
            for j in range(m):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
            W[k*n+i, col] = 1.0
            col += 1
        if col >= rank:
            break
    
    # Fill remaining
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.5
        V[:, col] = rng.randn(m*p) * 0.5
        W[:, col] = rng.randn(p*n) * 0.5
        col += 1
    
    return U, V, W

def _try_rank_als(n, m, p, rank, max_restarts=15, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using ALS with multiple initializations."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Choose initialization strategy
        strat = restart % 3
        if strat == 0:
            # Random
            scale = 0.5 + random.random() * 0.5
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale
        elif strat == 1:
            # Block-split structured
            U, V, W = _structured_init_block_split(n, m, p, rank, seed=seed + restart)
        else:
            # Strassen-inspired
            U, V, W = _structured_init_strassen_embed(n, m, p, rank, seed=seed + restart)
        
        # ALS iterations
        U, V, W = _als_iterate(U, V, W, n, m, p, max_iter=200)
        
        # Check error
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if max_err < 1e-8:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def _try_rank_gradient(n, m, p, rank, max_restarts=10, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using gradient descent."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5
        
        lr = 0.001
        prev_loss = float('inf')
        
        for step in range(2000):
            # Compute reconstruction
            T_rec = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            err = T_rec - T
            loss = np.sum(err**2)
            
            if loss < 1e-12:
                break
            if loss > 1e6:
                break
            
            # Gradients
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if loss < prev_loss * 0.99:
                lr *= 1.1
            elif loss > prev_loss * 1.01:
                lr *= 0.9
            prev_loss = loss
        
        # Check
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < 1e-8:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def _try_rank_scipy(n, m, p, rank, max_restarts=5, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using scipy.optimize."""
    from scipy.optimize import minimize
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    total_vars = rank * (dim_u + dim_v + dim_w)
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        x0 = np.random.randn(total_vars) * 0.3
        x0_u = x0[:rank*dim_u].reshape(dim_u, rank)
        x0_v = x0[rank*dim_u:rank*(dim_u+dim_v)].reshape(dim_v, rank)
        x0_w = x0[rank*(dim_u+dim_v):].reshape(dim_w, rank)
        
        def objective(x):
            U = x[:rank*dim_u].reshape(dim_u, rank)
            V = x[rank*dim_u:rank*(dim_u+dim_v)].reshape(dim_v, rank)
            W = x[rank*(dim_u+dim_v):].reshape(dim_w, rank)
            T_rec = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            return np.sum((T_rec - T)**2)
        
        try:
            res = minimize(objective, x0, method='BFGS', 
                          options={'maxiter': 500, 'gtol': 1e-8})
            if res.success and res.fun < 1e-10:
                U = res.x[:rank*dim_u].reshape(dim_u, rank)
                V = res.x[rank*dim_u:rank*(dim_u+dim_v)].reshape(dim_v, rank)
                W = res.x[rank*(dim_u+dim_v):].reshape(dim_w, rank)
                result = _try_snap_and_check(U, V, W, n, m, p)
                if result is not None:
                    return result
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    schoolbook_rank = n * m * p  # 48
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from lower to upper
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Realistically, try from 47 down
    
    # Phase 1: ALS with structured initializations
    for rank in range(47, 39, -1):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_als(n, m, p, rank, max_restarts=10, seed=rank * 1000, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # Phase 2: Gradient descent for a few ranks
    if best_result is None:
        for rank in range(45, 39, -1):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_gradient(n, m, p, rank, max_restarts=5, seed=rank * 2000, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    # Phase 3: SciPy optimization for a few ranks
    if best_result is None and time.time() < deadline - 15:
        for rank in range(44, 39, -1):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_scipy(n, m, p, rank, max_restarts=3, seed=rank * 3000, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
