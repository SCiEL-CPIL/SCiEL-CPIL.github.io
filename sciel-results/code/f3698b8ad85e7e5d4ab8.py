import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """
    Verify: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta_{j1, j2}
    U shape: (n*m, R)
    V shape: (m*p, R)
    W shape: (p*n, R)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    target = 1.0 if j1 == j2 else 0.0
                    # Sum over r
                    # W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :]
                    s = np.dot(W[k*n+i], U[i*m+j1] * V[j2*p+k])
                    if abs(s - target) > tol:
                        return False
    return True

def _als_step_U(U, V, W, n, m, p, R):
    """
    Update U given V, W.
    Equations: for all i, j1, k, j2:
    sum_r U[i*m+j1, r] * (W[k*n+i, r] * V[j2*p+k, r]) = delta_{j1, j2}
    
    For fixed i, j1, we solve for U[i*m+j1, :].
    The system is: A @ u = b
    where A has shape (p*m, R) and b has shape (p*m,).
    A[(k, j2), r] = W[k*n+i, r] * V[j2*p+k, r]
    b[(k, j2)] = 1.0 if j1 == j2 else 0.0
    """
    U_new = U.copy()
    for i in range(n):
        for j1 in range(m):
            # Build A and b
            A = np.zeros((p * m, R))
            b = np.zeros(p * m)
            
            for k in range(p):
                for j2 in range(m):
                    idx = k * m + j2
                    # A[idx, r] = W[k*n+i, r] * V[j2*p+k, r]
                    A[idx, :] = W[k*n+i, :] * V[j2*p+k, :]
                    b[idx] = 1.0 if j1 == j2 else 0.0
            
            # Solve least squares
            u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            U_new[i*m + j1, :] = u
            
    return U_new

def _als_step_V(U, V, W, n, m, p, R):
    """
    Update V given U, W.
    Equations: for all i, k, j1, j2:
    sum_r V[j2*p+k, r] * (W[k*n+i, r] * U[i*m+j1, r]) = delta_{j1, j2}
    
    For fixed k, j2, we solve for V[j2*p+k, :].
    System: A @ v = b
    A has shape (n*m, R), b has shape (n*m,).
    A[(i, j1), r] = W[k*n+i, r] * U[i*m+j1, r]
    b[(i, j1)] = 1.0 if j1 == j2 else 0.0
    """
    V_new = V.copy()
    for k in range(p):
        for j2 in range(m):
            A = np.zeros((n * m, R))
            b = np.zeros(n * m)
            
            for i in range(n):
                for j1 in range(m):
                    idx = i * m + j1
                    A[idx, :] = W[k*n+i, :] * U[i*m+j1, :]
                    b[idx] = 1.0 if j1 == j2 else 0.0
            
            v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            V_new[j2*p + k, :] = v
            
    return V_new

def _als_step_W(U, V, W, n, m, p, R):
    """
    Update W given U, V.
    Equations: for all i, k, j1, j2:
    sum_r W[k*n+i, r] * (U[i*m+j1, r] * V[j2*p+k, r]) = delta_{j1, j2}
    
    For fixed i, k, we solve for W[k*n+i, :].
    System: A @ w = b
    A has shape (m*m, R), b has shape (m*m,).
    A[(j1, j2), r] = U[i*m+j1, r] * V[j2*p+k, r]
    b[(j1, j2)] = 1.0 if j1 == j2 else 0.0
    """
    W_new = W.copy()
    for i in range(n):
        for k in range(p):
            A = np.zeros((m * m, R))
            b = np.zeros(m * m)
            
            for j1 in range(m):
                for j2 in range(m):
                    idx = j1 * m + j2
                    A[idx, :] = U[i*m+j1, :] * V[j2*p+k, :]
                    b[idx] = 1.0 if j1 == j2 else 0.0
            
            w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W_new[k*n + i, :] = w
            
    return W_new

def _compute_error(U, V, W, n, m, p):
    """Compute sum of squared errors."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    err = 0.0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    target = 1.0 if j1 == j2 else 0.0
                    s = np.dot(W[k*n+i], U[i*m+j1] * V[j2*p+k])
                    err += (s - target) ** 2
    return err

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    return U_snap, V_snap, W_snap

def _try_rank_als(n, m, p, target_rank, time_limit, num_restarts=5):
    """
    Try to find a decomposition of target_rank using ALS.
    """
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Random initialization
        # Scale matters: too large -> diverges, too small -> underfits
        # For rank R, typical magnitude is 1/sqrt(R) or similar.
        scale = 1.0 / np.sqrt(target_rank)
        U = np.random.randn(n*m, target_rank) * scale
        V = np.random.randn(m*p, target_rank) * scale
        W = np.random.randn(p*n, target_rank) * scale
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # ALS Iterations
        max_iters = 500
        for it in range(max_iters):
            if time.time() > start_time + time_limit:
                break
                
            U = _als_step_U(U, V, W, n, m, p, target_rank)
            V = _als_step_V(U, V, W, n, m, p, target_rank)
            W = _als_step_W(U, V, W, n, m, p, target_rank)
            
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Convergence check
            if err < 1e-12:
                break
                
        # Check if best solution can be snapped to half-integers
        if best_err < 1e-6:
            U_snap, V_snap, W_snap = _snap_to_half_integers(best_U, best_V, best_W)
            if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                
            # Try integer snap
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if _verify_exact(U_int, V_int, W_int, n, m, p):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
            # If close but not exact, try local refinement on snapped values
            # This is a heuristic: flip some entries to see if error reduces
            # But given the time constraint, we might skip complex local search here
            # and rely on more restarts.
            
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = 25
    # We will try ranks from schoolbook_rank - 1 down to 30
    # If we find one, we return it.
    
    # To maximize chance of finding a low rank, we should try lower ranks first?
    # No, usually it's easier to find a solution near the schoolbook rank, 
    # but the goal is MINIMAL rank.
    # However, if we start too low, we might fail.
    # Let's try a range.
    
    # Strategy: Try ranks 40 to 74.
    # If we find a valid one at 40, we return 40.
    # If we don't, we return 75.
    
    # Actually, we should try to find the MINIMAL rank.
    # So we should try 30, 31, ... 74?
    # But that takes too much time if many fail.
    # Better: Try a few low ranks (30-40), then if fail, try mid (40-60), then high.
    # Or just iterate downwards from 74?
    # If we iterate downwards, we return the FIRST valid one found, which is the highest rank we checked.
    # We want the LOWEST rank.
    # So we should iterate UPWARDS from a lower bound?
    # But we don't know the lower bound exactly.
    # Let's try a set of candidate ranks: [30, 35, 40, 45, 50, 55, 60, 65, 70, 74]
    # If we find one at 40, we return 40.
    # If we find one at 45, we return 45.
    # We should check in increasing order of rank to ensure minimality.
    
    candidate_ranks = list(range(30, schoolbook_rank))
    # Shuffle or sort? Sort by rank.
    # We want the smallest rank that works.
    
    # Time allocation:
    # Total time ~ 145s.
    # Number of ranks ~ 45.
    # Time per rank ~ 3s.
    # This is tight.
    
    # Let's prioritize lower ranks.
    # We'll try ranks 30 to 74.
    # If time runs out, return schoolbook.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # To save time, we can limit the number of restarts for lower ranks if they are likely to fail?
    # No, lower ranks are harder, so they need MORE time.
    
    # Let's just iterate.
    for target_rank in range(30, schoolbook_rank):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time: 5 seconds per rank?
        # 45 ranks * 5s = 225s. Too much.
        # Let's do 2 seconds per rank?
        # 45 * 2 = 90s. Okay.
        
        time_per_rank = 3.0
        # If we are near the end, reduce time.
        remaining_time = deadline - time.time()
        ranks_remaining = schoolbook_rank - target_rank
        if remaining_time / max(1, ranks_remaining) < time_per_rank:
             time_per_rank = max(0.5, remaining_time / max(1, ranks_remaining))
             
        result = _try_rank_als(n, m, p, target_rank, time_per_rank, num_restarts=3)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Final verification
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                return {"U": U_r, "V": V_r, "W": W_r}
                
    return schoolbook

# EVOLVE-BLOCK-END
