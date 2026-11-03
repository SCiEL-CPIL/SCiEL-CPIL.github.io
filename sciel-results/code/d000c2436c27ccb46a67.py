"""Matrix multiplication 2x3 x 3x4 rank-minimal decomposition.
Structural change: use a continuous alternating-minimization (ALS) solver with
random restarts, then snap to half-integers and verify exactly. This replaces
the discrete random-perturbation search with a fundamentally different
optimization family (nonlinear least-squares on the tensor equations).
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p, tol=1e-8):
    """Exact verification of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            val = np.dot(U[i * m + j], V[jp * p + k] * W[kp * n + ip])
                            if abs(val - expected) > tol:
                                return False
    return True

def _snap_to_half_int(arr):
    return np.round(np.asarray(arr, dtype=float) * 2) / 2.0

def _is_half_integer(arr, tol=1e-7):
    arr = np.asarray(arr, dtype=float)
    return np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol)

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

def _tensor_error(U, V, W, n, m, p):
    """Compute the total squared error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            val = np.dot(U[i * m + j], V[jp * p + k] * W[kp * n + ip])
                            err += (val - expected) ** 2
    return err

def _als_step_U(U, V, W, n, m, p, rank):
    """Solve for U given V, W via least squares."""
    new_U = U.copy()
    for i in range(n):
        for j in range(m):
            # For each k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                new_U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
            except:
                pass
    return new_U

def _als_step_V(U, V, W, n, m, p, rank):
    """Solve for V given U, W via least squares."""
    new_V = V.copy()
    for j in range(m):
        for k in range(p):
            # For each i: sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                new_V[j * p + k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
            except:
                pass
    return new_V

def _als_step_W(U, V, W, n, m, p, rank):
    """Solve for W given U, V via least squares."""
    new_W = W.copy()
    for k in range(p):
        for i in range(n):
            # For each j: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                new_W[k * n + i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
            except:
                pass
    return new_W

def _als_solve(n, m, p, rank, U, V, W, max_iter=500, tol=1e-12, deadline=None):
    """Run alternating least squares to minimize tensor error."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        U = _als_step_U(U, V, W, n, m, p, rank)
        V = _als_step_V(U, V, W, n, m, p, rank)
        W = _als_step_W(U, V, W, n, m, p, rank)
        err = _tensor_error(U, V, W, n, m, p)
        if err < tol:
            return U, V, W, True
    return U, V, W, False

def _try_rank_als(n, m, p, rank, n_restarts, deadline, rng):
    """Try to find a rank-r decomposition via ALS with random restarts."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            return None
        
        # Random initialization with small values
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        # Run ALS
        U, V, W, converged = _als_solve(n, m, p, rank, U, V, W, max_iter=300, tol=1e-12, deadline=deadline)
        
        if converged:
            # Snap to half-integers
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            
            if _verify_exact(U_s, V_s, W, n, m, p):
                if _is_half_integer(U_s) and _is_half_integer(V_s) and _is_half_integer(W_s):
                    return (U_s, V_s, W_s)
            
            # Also try the unsnapped version if it's close enough
            if _verify_exact(U, V, W, n, m, p, tol=1e-6):
                U_s = _snap_to_half_int(U)
                V_s = _snap_to_half_int(V)
                W_s = _snap_to_half_int(W)
                if _verify_exact(U_s, V_s, W_s, n, m, p):
                    return (U_s, V_s, W_s)
        
        # Also try: if error is small, snap and check
        err = _tensor_error(U, V, W, n, m, p)
        if err < 1e-6:
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return (U_s, V_s, W_s)
    
    return None

def _try_rank_schoolbook_merge(n, m, p, rank, deadline, rng):
    """Try to find a lower-rank decomposition by starting from schoolbook
    and using ALS to merge columns."""
    if rank >= n * m * p:
        return None
    
    # Start from schoolbook and zero out some columns
    U, V, W = _schoolbook(n, m, p)
    full_rank = n * m * p
    
    # Randomly zero out (full_rank - rank) columns
    cols_to_zero = rng.sample(range(full_rank), full_rank - rank)
    for c in cols_to_zero:
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0
    
    # Trim to target rank
    keep_cols = [c for c in range(full_rank) if c not in cols_to_zero]
    U = U[:, keep_cols]
    V = V[:, keep_cols]
    W = W[:, keep_cols]
    
    # Run ALS to try to recover the tensor
    U, V, W, converged = _als_solve(n, m, p, rank, U, V, W, max_iter=500, tol=1e-12, deadline=deadline)
    
    if converged or _tensor_error(U, V, W, n, m, p) < 1e-6:
        U_s = _snap_to_half_int(U)
        V_s = _snap_to_half_int(V)
        W_s = _snap_to_half_int(W)
        if _verify_exact(U_s, V_s, W_s, n, m, p):
            return (U_s, V_s, W_s)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    np.random.seed(42)
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_fb, V_fb, W_fb)
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Try ranks from low to high
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Method 1: ALS with random restarts
        result = _try_rank_als(n, m, p, target_rank, n_restarts=20, deadline=deadline - 10, rng=rng)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = (U, V, W)
            break
        
        # Method 2: Schoolbook merge with ALS
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            result = _try_rank_schoolbook_merge(n, m, p, target_rank, deadline - 10, rng)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_result = (U, V, W)
                break
        
        if best_rank == target_rank:
            break
    
    # Return best found
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
