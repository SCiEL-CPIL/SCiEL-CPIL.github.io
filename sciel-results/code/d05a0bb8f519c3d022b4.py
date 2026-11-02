import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def schoolbook(n, m, p):
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

def reconstruct(U, V, W):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for c in range(r):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return T

def objective(params, T_flat, n, m, p, r):
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    T_rec = reconstruct(U, V, W)
    diff = T_rec.flatten() - T_flat
    return np.dot(diff, diff)

def verify(U, V, W, T):
    T_rec = reconstruct(U, V, W)
    return np.allclose(T_rec, T, atol=1e-9)

def try_find_rank(T, r, time_limit):
    n, m, p = DIMENSIONS
    T_flat = T.flatten()
    total_vars = n*m*r + m*p*r + p*n*r
    
    best_error = np.inf
    best_params = None
    
    start = time.time()
    attempts = 0
    # Limit attempts to ensure we don't run too long if one attempt is slow
    max_attempts = 30
    
    while time.time() - start < time_limit and attempts < max_attempts:
        np.random.seed(attempts)
        # Initialize with small random values
        init = np.random.randn(total_vars) * 0.5
        
        # Use L-BFGS-B for optimization
        try:
            res = minimize(objective, init, args=(T_flat, n, m, p, r), 
                           method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
            if best_error < 1e-8:
                break
        except Exception:
            pass
        attempts += 1
        
    if best_params is None or best_error > 1e-4:
        return None
    
    # Extract U, V, W
    U = best_params[:n*m*r].reshape(n*m, r)
    V = best_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = best_params[n*m*r + m*p*r :].reshape(p*n, r)
    
    # The task requires integer or half-integer entries.
    # Try rounding to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if verify(U_r, V_r, W_r, T):
        return U_r, V_r, W_r
        
    # Try rounding to nearest integer
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if verify(U_i, V_i, W_i, T):
        return U_i, V_i, W_i
        
    # Try rounding to nearest 1/3 (just in case, though unlikely for this problem)
    U_t = np.round(U * 3) / 3
    V_t = np.round(V * 3) / 3
    W_t = np.round(W * 3) / 3
    if verify(U_t, V_t, W_t, T):
        return U_t, V_t, W_t

    return None

def construct():
    n, m, p = DIMENSIONS
    # Build target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
        
    # Fallback solution (Rank 18)
    U_fb, V_fb, W_fb = schoolbook(n, m, p)
    
    start_time = time.time()
    # We have 120s total. Reserve 10s for safety.
    # Try Rank 16 first.
    time_limit_16 = 60
    result_16 = try_find_rank(T, 16, time_limit_16)
    
    if result_16:
        U, V, W = result_16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If Rank 16 fails, try Rank 17 with remaining time
    elapsed = time.time() - start_time
    remaining = 100 - elapsed
    if remaining > 10:
        result_17 = try_find_rank(T, 17, remaining)
        if result_17:
            U, V, W = result_17
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 18
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
