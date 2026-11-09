import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _solve_for_W(U, V, n, m, p, R):
    """Given U and V, solve linear system for W exactly."""
    # Variables: W[k*n+i, r] for k in range(p), i in range(n), r in range(R)
    # Equations: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    num_eq = n * m * p
    num_var = p * n * R
    
    A = np.zeros((num_eq, num_var))
    b = np.ones(num_eq)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    var_idx = (k * n + i) * R + r
                    A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                eq_idx += 1
    
    # Check if system is consistent and find exact solution
    # Use rank-revealing approach
    rank_A = np.linalg.matrix_rank(A, tol=1e-10)
    rank_aug = np.linalg.matrix_rank(np.column_stack([A, b]), tol=1e-10)
    
    if rank_A != rank_aug:
        return None
    
    # Find a solution - use least squares and check
    W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=1e-10)
    
    # Check exactness
    if not np.allclose(A @ W_flat, b, atol=1e-9):
        return None
    
    W = W_flat.reshape((p * n, R))
    return W

def _try_decomposition(n, m, p, R, U_init, V_init):
    """Try to complete U, V to a full decomposition by solving for W."""
    U = U_init.copy()
    V = V_init.copy()
    
    W = _solve_for_W(U, V, n, m, p, R)
    if W is None:
        return None
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Verify with snapped values
    if not _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
        return None
    
    return U_snap, V_snap, W_snap

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Candidate ranks to try (in increasing order)
    candidates = sorted(set([
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95
    ]))
    
    for R in candidates:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
        
        found = False
        
        # Try multiple random initializations
        max_attempts = 50 if R <= 40 else 30 if R <= 60 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random small integer initialization
            U_init = np.random.randint(-1, 2, (n*m, R))
            V_init = np.random.randint(-1, 2, (m*p, R))
            
            result = _try_decomposition(n, m, p, R, U_init, V_init)
            if result is not None:
                U_s, V_s, W_s = result
                best_rank = R
                best = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                found = True
                break
        
        # Also try structured initializations
        if not found:
            # Pattern: U[i*m+j, r] = 1 if r < j else 0, V[j*p+k, r] = 1 if r < k else 0
            # This is a triangular-like structure
            U_struct = np.zeros((n*m, R))
            V_struct = np.zeros((m*p, R))
            for i in range(n):
                for j in range(m):
                    for r in range(min(j+1, R)):
                        U_struct[i*m+j, r] = 1
            for j in range(m):
                for k in range(p):
                    for r in range(min(k+1, R)):
                        V_struct[j*p+k, r] = 1
            
            result = _try_decomposition(n, m, p, R, U_struct, V_struct)
            if result is not None:
                U_s, V_s, W_s = result
                best_rank = R
                best = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                found = True
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
    
    return best
