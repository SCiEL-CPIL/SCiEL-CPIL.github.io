import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct Schoolbook Decomposition (Rank n*m*p)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook (should be exact)
    # It is exact by construction.
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # 2. Attempt to find lower rank (e.g., 47)
    # We try ranks from best_rank - 1 down to a lower bound
    # Lower bound is at least max(n, m, p) = 4, but realistically much higher.
    # Let's try 47, 46, ... down to 40 within time limit.
    
    for rank in range(best_rank - 1, max(n, m, p) - 1, -1):
        if time.time() > deadline - 5:
            break
            
        found = _try_find_rank(n, m, p, rank, max_restarts=20, time_limit=5)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            break
            
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=10):
    """
    Attempts to find a rank-r exact decomposition with half-integer entries.
    Uses ALS with vectorized updates and snapping.
    """
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # ALS Iterations
        for it in range(50):
            if time.time() > start_time + time_limit:
                break
                
            # Update U
            # U[idx, :] is solved from A @ U[idx, :] = b
            # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            # This is hard to vectorize fully across all idx because i,j depend on idx.
            # But we can loop over idx and use matrix ops for the rank dimension.
            
            U_new = np.zeros_like(U)
            for idx in range(n * m):
                i, j = divmod(idx, m)
                # A is (p, rank)
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # We can form A by broadcasting
                # V_rows = V[j*p:(j+1)*p, :] # (p, rank)
                # W_rows = W[np.arange(p)*n + i, :] # (p, rank)
                # A = V_rows * W_rows
                V_part = V[j*p:(j+1)*p, :]
                W_part = W[np.arange(p)*n + i, :]
                A = V_part * W_part
                b = np.ones(p)
                # Solve A @ x = b
                try:
                    # Using lstsq for robustness
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except:
                    U_new[idx, :] = U[idx, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for idx in range(m * p):
                j, k = divmod(idx, p)
                # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                U_part = U[np.arange(n)*m + j, :] # (n, rank)
                W_part = W[k*n:(k+1)*n, :] # (n, rank)
                A = U_part * W_part
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except:
                    V_new[idx, :] = V[idx, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for idx in range(p * n):
                k, i = divmod(idx, n)
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                U_part = U[i*m:(i+1)*m, :] # (m, rank)
                V_part = V[np.arange(m)*p + k, :] # (m, rank)
                A = U_part * V_part
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except:
                    W_new[idx, :] = W[idx, :]
            W = W_new
            
        # Check error
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Try to snap
            U_snapped, u_ok = _snap(U, 1e-5)
            V_snapped, v_ok = _snap(V, 1e-5)
            W_snapped, w_ok = _snap(W, 1e-5)
            
            if u_ok and v_ok and w_ok:
                # Verify exactness with snapped values
                if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                    
    return None

def _snap(arr, tol=1e-6):
    """Snap to nearest half-integer if close."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
