"""Optimized matrix multiplication tensor decomposition for 3x5 x 5x5."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
_START_TIME = time.time()
_TIME_LIMIT = 120

# EVOLVE-BLOCK-START

def _verify_decomp(U, V, W, n, m, p, tol=1e-6):
    """Verify: for all i,k,j1,j2: sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] == delta_{j1,j2}"""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Build the full tensor check efficiently
    for i in range(n):
        for k in range(p):
            # For this (i,k), we need a 5x5 matrix M where M[j1,j2] = sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r]
            # M should be identity
            M = np.zeros((m, m))
            for j1 in range(m):
                u_row = U[i*m+j1, :]  # shape (r,)
                for j2 in range(m):
                    v_row = V[j2*p+k, :]  # shape (r,)
                    M[j1, j2] = np.dot(u_row, W[k*n+i, :] * v_row)
            # Check M == I
            if not np.allclose(M, np.eye(m), atol=tol):
                return False
    return True

def _verify_decomp_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using matrix operations."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            # M[j1,j2] = sum_r W[k*n+i,r] * U[i*m+j1,r] * V[j2*p+k,r]
            # = (U[i*m:(i+1)*m, :] * W[k*n+i, :]) @ (V[:, k] * W[k*n+i, :])^T ... no
            # Actually: let a_j1 = U[i*m+j1, :], b_j2 = V[j2*p+k, :], w = W[k*n+i, :]
            # M[j1,j2] = sum_r w_r * a_j1_r * b_j2_r = (a_j1 * w) @ b_j2
            # So M = diag(a_0 * w) ... no, M = (U[i*m:(i+1)*m, :] * w) @ (V[:, k] * w)^T? 
            # No: M[j1,j2] = sum_r (U[i*m+j1,r] * w_r) * V[j2*p+k, r]
            # Let A' = U[i*m:(i+1)*m, :] * w  (shape m x r)
            # Let B' = V[:, k] * w  (shape m x r) -- but V is indexed by (j2*p+k)
            # V[j2*p+k, r] for j2 in range(m) is V[k::p, r] if V is stored as (j*p+k)
            # Wait, V has shape (m*p, r), indexed by j2*p+k
            # So for fixed k, V[j2*p+k, :] for j2=0..m-1 is rows k, k+p, k+2p, ...
            V_sub = V[k::p, :]  # shape (m, r), V_sub[j2, r] = V[j2*p+k, r]
            A_sub = U[i*m:(i+1)*m, :]  # shape (m, r)
            w = W[k*n+i, :]  # shape (r,)
            M = (A_sub * w) @ V_sub.T  # shape (m, m)
            if not np.allclose(M, np.eye(m), atol=tol):
                return False
    return True

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are integers or half-integers."""
    M2 = np.asarray(M, dtype=float) * 2.0
    return np.all(np.abs(M2 - np.round(M2)) < tol)

def _snap_to_half(M):
    """Snap matrix to nearest half-integer."""
    return np.round(np.asarray(M, dtype=float) * 2.0) / 2.0

def _als_optimize(U, V, W, n, m, p, r, max_iters=100, deadline=None):
    """ALS optimization. Returns optimized (U, V, W)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m*m, r))
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                b = np.zeros(m*m)
                for j in range(m):
                    b[j*m+j] = 1.0
                try:
                    w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass
        
        # Update U
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                try:
                    u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                    U[i*m+j1, :] = u
                except:
                    pass
        
        # Update V
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                try:
                    v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                    V[j2*p+k, :] = v
                except:
                    pass
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute Frobenius norm of residual."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    total_err = 0.0
    for i in range(n):
        for k in range(p):
            V_sub = V[k::p, :]
            A_sub = U[i*m:(i+1)*m, :]
            w = W[k*n+i, :]
            M = (A_sub * w) @ V_sub.T
            total_err += np.sum((M - np.eye(m))**2)
    return total_err

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT - 10  # safety margin
    
    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomp_fast(best_U, best_V, best_W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    np.random.seed(42)
    
    # Strategy: Try to reduce rank from 75 down to lower values
    # For each candidate rank r, try multiple random initializations with ALS
    
    # Try ranks from 50 up to 74 (we know 75 works)
    # Focus on promising ranges based on known results
    
    candidate_ranks = list(range(50, 75))
    
    for r in candidate_ranks:
        if time.time() > deadline - 3:
            break
        
        # Try several random initializations
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            # Initialize with sparse random half-integers
            U_init = np.zeros((n*m, r))
            V_init = np.zeros((m*p, r))
            W_init = np.zeros((p*n, r))
            
            # Random sparse initialization
            for col in range(r):
                # Each column has a few non-zero entries
                nnz = np.random.randint(1, 4)
                # U entries
                u_rows = np.random.choice(n*m, size=min(nnz, n*m), replace=False)
                U_init[u_rows, col] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nnz)
                # V entries
                v_rows = np.random.choice(m*p, size=min(nnz, m*p), replace=False)
                V_init[v_rows, col] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nnz)
                # W entries
                w_rows = np.random.choice(p*n, size=min(nnz, p*n), replace=False)
                W_init[w_rows, col] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nnz)
            
            # Run ALS optimization
            U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_init, n, m, p, r, 
                                                   max_iters=50, deadline=deadline-3)
            
            # Check if close to valid
            residual = _compute_residual(U_opt, V_opt, W_opt, n, m, p)
            
            if residual < 1e-4:
                # Try snapping to half-integers
                U_snap = _snap_to_half(U_opt)
                V_snap = _snap_to_half(V_opt)
                W_snap = _snap_to_half(W_opt)
                
                if _verify_decomp_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-4):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    # Found a valid decomposition at this rank
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            # Also try without snapping (in case ALS found exact solution)
            if _verify_decomp_fast(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                # Check if half-integer
                if _is_half_integer_matrix(U_opt) and _is_half_integer_matrix(V_opt) and _is_half_integer_matrix(W_opt):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If no improvement found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
