"""Tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=8, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _solve_for_W(U, V, n, m, p):
    """Given U and V, solve for W using least squares."""
    R = U.shape[1]
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            b = np.ones(m)
            if R > 0:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = w
    return W

def _solve_for_V(U, W, n, m, p):
    """Given U and W, solve for V using least squares."""
    R = U.shape[1]
    V = np.zeros((m * p, R))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            b = np.ones(n)
            if R > 0:
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = v
    return V

def _solve_for_U(V, W, n, m, p):
    """Given V and W, solve for U using least squares."""
    R = V.shape[1]
    U = np.zeros((n * m, R))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            b = np.ones(p)
            if R > 0:
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = u
    return U

def _round_to_half_int(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum error in the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing one column and re-solving."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    for r in range(rank):
        if time.time() > deadline:
            break
        
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        U_trial = U[:, mask].copy()
        V_trial = V[:, mask].copy()
        W_trial = W[:, mask].copy()
        
        for _ in range(15):
            if time.time() > deadline:
                break
            W_trial = _solve_for_W(U_trial, V_trial, n, m, p)
            V_trial = _solve_for_V(U_trial, W_trial, n, m, p)
            U_trial = _solve_for_U(V_trial, W_trial, n, m, p)
        
        U_r = _round_to_half_int(U_trial)
        V_r = _round_to_half_int(V_trial)
        W_r = _round_to_half_int(W_trial)
        
        if _validate(U_r, V_r, W_r, n, m, p):
            return (U_r, V_r, W_r)
    
    return None

def _initialize_structured(n, m, p, target_rank, seed=0):
    """Initialize with structured patterns that exploit the tensor geometry."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Strategy: distribute terms across (i,j,k) with some structure
    for c in range(target_rank):
        # Use a cyclic pattern
        i = c % n
        j = (c // n) % m
        k = (c // (n * m)) % p
        
        # Add variation
        if c < target_rank // 2:
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        else:
            # Second half: use different (i,j,k) combinations
            i2 = (c + 1) % n
            j2 = (c // n + 1) % m
            k2 = (c // (n * m) + 1) % p
            U[i2 * m + j2, c] = 1.0
            V[j2 * p + k2, c] = 1.0
            W[k2 * n + i2, c] = 1.0
    
    # Add small random perturbation to escape symmetry
    U += rng.randn(n * m, target_rank) * 0.05
    V += rng.randn(m * p, target_rank) * 0.05
    W += rng.randn(p * n, target_rank) * 0.05
    
    return U, V, W

def _alternating_minimize(U, V, W, n, m, p, max_iter=200, deadline=None, tol=1e-10):
    """Run alternating minimization with convergence check."""
    prev_err = float('inf')
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        W = _solve_for_W(U, V, n, m, p)
        V = _solve_for_V(U, W, n, m, p)
        U = _solve_for_U(V, W, n, m, p)
        
        err = _compute_error(U, V, W, n, m, p)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s limit
    
    # Start with schoolbook rank 72
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
    
    # Try to reduce from schoolbook
    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    while time.time() < deadline - 30:
        result = _try_reduce_rank(U, V, W, n, m, p, deadline - 30)
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            break
    
    # Try various target ranks with multiple initializations
    # Focus on ranks that are plausible improvements
    target_ranks = list(range(24, best_rank))
    
    for target_rank in reversed(target_ranks):
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            U_trial, V_trial, W_trial = _initialize_structured(n, m, p, target_rank, seed)
            U_trial, V_trial, W_trial, err = _alternating_minimize(
                U_trial, V_trial, W_trial, n, m, p, 
                max_iter=300, deadline=deadline - 10
            )
            
            U_r = _round_to_half_int(U_trial)
            V_r = _round_to_half_int(V_trial)
            W_r = _round_to_half_int(W_trial)
            
            if _validate(U_r, V_r, W_r, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                found = True
                break
        
        if found:
            # Try to reduce further from this solution
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            while time.time() < deadline - 10:
                result = _try_reduce_rank(U, V, W, n, m, p, deadline - 10)
                if result is not None:
                    U, V, W = result
                    if U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
