"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check half-integer constraint
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
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

def _try_rank_altsolve(n, m, p, rank, max_iter=500, seed=None, tol=1e-10):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        Ua_snapped = np.vectorize(_snap_to_half)(best_U)
        Va_snapped = np.vectorize(_snap_to_half)(best_V)
        Wa_snapped = np.vectorize(_snap_to_half)(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_gradient(n, m, p, rank, max_iter=3000, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    lr = 0.05
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        error = np.max(np.abs(err))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if error < 1e-10:
            break
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        
        U3 -= lr * grad_U
        V3 -= lr * grad_V
        W3 -= lr * grad_W
        
        if it % 100 == 0:
            lr *= 0.8
    
    Ua = best_U.reshape(n*m, rank)
    Va = best_V.reshape(m*p, rank)
    Wa = best_W.reshape(p*n, rank)
    
    Ua_snapped = np.vectorize(_snap_to_half)(Ua)
    Va_snapped = np.vectorize(_snap_to_half)(Va)
    Wa_snapped = np.vectorize(_snap_to_half)(Wa)
    
    U_out = Ua_snapped.tolist()
    V_out = Va_snapped.tolist()
    W_out = Wa_snapped.tolist()
    
    if _verify_fast(U_out, V_out, W_out, n, m, p):
        return U_out, V_out, W_out
    
    U_out_raw = Ua.tolist()
    V_out_raw = Va.tolist()
    W_out_raw = Wa.tolist()
    
    if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
        return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = Va[j*p + k] * Wa[k*n + i]
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = Ua[i*m + j] * Wa[k*n + i]
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = Ua[i*m + j] * Va[j*p + k]
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _try_low_rank_exact(n, m, p, rank, max_iter=200, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    for init_type in range(4):
        if _remaining() < 5.0:
            break
        
        if init_type == 0:
            U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
            V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
            W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        elif init_type == 1:
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, r] = rng.uniform(0.5, 1.5)
                V[j*p+k, r] = rng.uniform(0.5, 1.5)
                W[k*n+i, r] = rng.uniform(0.5, 1.5)
                if rng.random() < 0.3:
                    i2 = rng.randint(0, n-1)
                    j2 = rng.randint(0, m-1)
                    k2 = rng.randint(0, p-1)
                    U[i2*m+j2, r] += rng.uniform(-0.5, 0.5)
                    V[j2*p+k2, r] += rng.uniform(-0.5, 0.5)
                    W[k2*n+i2, r] += rng.uniform(-0.5, 0.5)
        elif init_type == 2:
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                comp_type = rng.randint(0, 3)
                if comp_type == 0:
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j, r] = rng.uniform(0.5, 1.5)
                    V[j*p+k, r] = rng.uniform(0.5, 1.5)
                    W[k*n+i, r] = rng.uniform(0.5, 1.5)
                elif comp_type == 1:
                    i = rng.randint(0, n-1)
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j, r] += rng.uniform(0.1, 0.5)
                            V[j*p+k, r] += rng.uniform(0.1, 0.5)
                            W[k*n+i, r] += rng.uniform(0.1, 0.5)
                elif comp_type == 2:
                    j = rng.randint(0, m-1)
                    for i in range(n):
                        for k in range(p):
                            U[i*m+j, r] += rng.uniform(0.1, 0.5)
                            V[j*p+k, r] += rng.uniform(0.1, 0.5)
                            W[k*n+i, r] += rng.uniform(0.1, 0.5)
                else:
                    k = rng.randint(0, p-1)
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, r] += rng.uniform(0.1, 0.5)
                            V[j*p+k, r] += rng.uniform(0.1, 0.5)
                            W[k*n+i, r] += rng.uniform(0.1, 0.5)
        else:
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                for _ in range(rng.randint(1, 3)):
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j, r] = rng.choice([-1, 1])
                    V[j*p+k, r] = rng.choice([-1, 1])
                    W[k*n+i, r] = rng.choice([-1, 1])
        
        for it in range(max_iter):
            if _remaining() < 3.0:
                break
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p + k] * W[k*n + i]
                    try:
                        U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m + j] * W[k*n + i]
                    try:
                        V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m + j] * V[j*p + k]
                    try:
                        W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-10:
                U_snapped = np.vectorize(_snap_to_half)(U).tolist()
                V_snapped = np.vectorize(_snap_to_half)(V).tolist()
                W_snapped = np.vectorize(_snap_to_half)(W).tolist()
                if _verify_fast(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                # If snapping fails, the solution is not on the lattice, so it's invalid for this task
    
    return None

def _try_scipy_optimize(n, m, p, rank, max_iter=100, seed=None):
    """
    Use scipy.optimize.minimize with a smooth objective (sum of squared errors)
    to find a low-rank decomposition. This is a different approach from the
    alternating least squares used elsewhere.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    def _flatten_params(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:2*n*m*rank]
        W_flat = params[2*n*m*rank:3*n*m*rank]
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        return U, V, W
    
    def _objective(params):
        U, V, W = _flatten_params(params)
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        return np.sum((T - 1.0)**2)
    
    def _gradient(params):
        U, V, W = _flatten_params(params)
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = 2 * (T - 1.0)
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V, W)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U, W)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    # Try multiple initializations
    for init_seed in range(5):
        if _remaining() < 10.0:
            break
        
        rng_init = random.Random(seed + init_seed)
        U_init = np.array([[rng_init.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
        V_init = np.array([[rng_init.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
        W_init = np.array([[rng_init.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(
                _objective,
                x0,
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except:
            continue
        
        if result.success or _remaining() < 5.0:
            U, V, W = _flatten_params(result.x)
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-9:
                U_snapped = np.vectorize(_snap_to_half)(U3).tolist()
                V_snapped = np.vectorize(_snap_to_half)(V3).tolist()
                W_snapped = np.vectorize(_snap_to_half)(W3).tolist()
                if _verify_fast(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                # If snapping fails, the solution is not on the lattice, so it's invalid for this task
                # Do not return the raw solution
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    min_rank = max(n*m, m*p, n*p)
    
    # Try ranks from min_rank up to best_rank - 1
    # Allocate time to each rank based on remaining time
    # We prioritize lower ranks but ensure we have time to try higher ones if lower ones fail
    
    # First, try min_rank with high effort
    for attempt in range(10):
        if _remaining() < 15.0:
            break
        
        result = _try_scipy_optimize(n, m, p, min_rank, max_iter=200, seed=min_rank*100+attempt)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = min_rank
                break
    
    if best_rank == n * m * p:
        for attempt in range(10):
            if _remaining() < 10.0:
                break
            
            result = _try_low_rank_exact(n, m, p, min_rank, max_iter=300, seed=min_rank*200+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = min_rank
                    break

    # If still not found, try higher ranks
    if best_rank == n * m * p:
        # Search ranks from min_rank + 1 to best_rank - 1
        # Use a time-based loop
        for rank in range(min_rank + 1, best_rank):
            if _remaining() < 10.0:
                break
            
            # Try a few attempts for each rank
            # Reduce attempts as rank increases if time is tight
            attempts = 5 if rank < 40 else 3 if rank < 60 else 2
            
            found = False
            for attempt in range(attempts):
                if _remaining() < 5.0:
                    break
                
                result = _try_scipy_optimize(n, m, p, rank, max_iter=150, seed=rank*100+attempt)
                if result is not None:
                    U, V, W = result
                    if _verify_fast(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                break
    
    # Final check
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
