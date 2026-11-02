"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.max(np.abs(T - 1.0)) < tol

def _max_error(U, V, W, n, m, p):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.max(np.abs(T - 1.0))

def _schoolbook(n, m, p):
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

def _local_search(U, V, W, n, m, p, max_iters=2000, time_limit=10.0, start_time=None):
    """Local search: try changing individual entries to reduce max error."""
    if start_time is None:
        start_time = time.time()
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Precompute contributions for efficiency
    # For each (i,j,k), T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # If we change U[i0,j0,r0] by delta, the change in T[i,j,k] is:
    #   delta * V[j0,k,r0] * W[k,i0,r0] for all (i,j0,k) where j=j0
    # More precisely, T[i,j0,k] changes by delta * V[j0,k,r0] * W[k,i,r0]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _max_error(best_U, best_V, best_W, n, m, p)
    
    values = [0.0, 0.5, 1.0, -0.5, -1.0]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if best_err < 1e-10:
            break
            
        # Recompute current errors
        U_r = best_U.reshape(n, m, rank)
        V_r = best_V.reshape(m, p, rank)
        W_r = best_W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        errors = np.abs(T - 1.0)
        
        # Find worst (i,j,k)
        worst = np.unravel_index(np.argmax(errors), errors.shape)
        wi, wj, wk = worst
        
        # Try adjusting U[wi*m+wj, r], V[wj*p+wk, r], W[wk*n+wi, r]
        ui = wi * m + wj
        vj = wj * p + wk
        wk_idx = wk * n + wi
        
        improved = False
        
        # Try U changes
        for r in range(rank):
            old_u = best_U[ui, r]
            for nv in values:
                if nv == old_u:
                    continue
                best_U[ui, r] = nv
                err = _max_error(best_U, best_V, best_W, n, m, p)
                if err < best_err - 1e-12:
                    best_err = err
                    improved = True
                    break
                else:
                    best_U[ui, r] = old_u
            if improved:
                break
        
        if improved:
            continue
        
        # Try V changes
        for r in range(rank):
            old_v = best_V[vj, r]
            for nv in values:
                if nv == old_v:
                    continue
                best_V[vj, r] = nv
                err = _max_error(best_U, best_V, best_W, n, m, p)
                if err < best_err - 1e-12:
                    best_err = err
                    improved = True
                    break
                else:
                    best_V[vj, r] = old_v
            if improved:
                break
        
        if improved:
            continue
        
        # Try W changes
        for r in range(rank):
            old_w = best_W[wk_idx, r]
            for nv in values:
                if nv == old_w:
                    continue
                best_W[wk_idx, r] = nv
                err = _max_error(best_U, best_V, best_W, n, m, p)
                if err < best_err - 1e-12:
                    best_err = err
                    improved = True
                    break
                else:
                    best_W[wk_idx, r] = old_w
            if improved:
                break
        
        if not improved:
            # Stuck, try random perturbation
            rng = np.random.RandomState(it)
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = rng.randint(n*m)
                r = rng.randint(rank)
                best_U[idx, r] = rng.choice(values)
            elif mat == 'V':
                idx = rng.randint(m*p)
                r = rng.randint(rank)
                best_V[idx, r] = rng.choice(values)
            else:
                idx = rng.randint(p*n)
                r = rng.randint(rank)
                best_W[idx, r] = rng.choice(values)
            new_err = _max_error(best_U, best_V, best_W, n, m, p)
            if new_err > best_err + 0.1:
                # Revert if too bad
                if mat == 'U':
                    best_U[idx, r] = 0.0
                elif mat == 'V':
                    best_V[idx, r] = 0.0
                else:
                    best_W[idx, r] = 0.0
    
    return best_U, best_V, best_W, best_err

def _try_reduce_rank(U, V, W, n, m, p, time_limit=15.0):
    """Try to reduce rank by zeroing out columns and verifying."""
    start_time = time.time()
    rank = U.shape[1]
    U, V, W, err = _local_search(U, V, W, n, m, p, max_iters=500, time_limit=min(5, time_limit), start_time=start_time)
    
    # Try to zero out columns one by one
    cols_to_try = list(range(rank))
    rng = np.random.RandomState(42)
    rng.shuffle(cols_to_try)
    
    for col in cols_to_try:
        if time.time() - start_time > time_limit:
            break
        if err < 1e-10:
            break
        
        # Try zeroing this column
        U_save, V_save, W_save = U[:, col].copy(), V[:, col].copy(), W[:, col].copy()
        U[:, col] = 0
        V[:, col] = 0
        W[:, col] = 0
        
        new_err = _max_error(U, V, W, n, m, p)
        if new_err < 1e-8:
            # Column is redundant, keep it zero
            err = new_err
        else:
            # Restore
            U[:, col] = U_save
            V[:, col] = V_save
            W[:, col] = W_save
    
    # Final local search
    U, V, W, err = _local_search(U, V, W, n, m, p, max_iters=1000, time_limit=time_limit - (time.time() - start_time), start_time=start_time)
    
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try local search to find a valid decomposition with same rank
    U_ls, V_ls, W_ls, err = _local_search(U, V, W, n, m, p, max_iters=1000, time_limit=10)
    if err < 1e-8:
        best_U, best_V, best_W = U_ls, V_ls, W_ls
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        U_red, V_red, W_red, err_red = _try_reduce_rank(best_U, best_V, best_W, n, m, p, time_limit=20)
        if err_red < 1e-8:
            # Count non-zero columns
            active_cols = 0
            for r in range(U_red.shape[1]):
                if np.any(U_red[:, r] != 0) or np.any(V_red[:, r] != 0) or np.any(W_red[:, r] != 0):
                    active_cols += 1
            if active_cols < best_rank:
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = active_cols
    
    # Try random restarts with different initializations
    if time.time() < deadline - 30:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random sparse initialization
            rng = np.random.RandomState(100 + trial)
            target_rank = min(80, best_rank - 5)
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # Sparse random
                u_idx = rng.choice(n*m, size=rng.randint(2, 6), replace=False)
                v_idx = rng.choice(m*p, size=rng.randint(2, 6), replace=False)
                w_idx = rng.choice(p*n, size=rng.randint(2, 6), replace=False)
                for idx in u_idx:
                    U_r[idx, r] = rng.choice([0.5, 1.0, -0.5, -1.0])
                for idx in v_idx:
                    V_r[idx, r] = rng.choice([0.5, 1.0, -0.5, -1.0])
                for idx in w_idx:
                    W_r[idx, r] = rng.choice([0.5, 1.0, -0.5, -1.0])
            
            U_ls2, V_ls2, W_ls2, err2 = _local_search(U_r, V_r, W_r, n, m, p, max_iters=500, time_limit=8)
            if err2 < 1e-8:
                active = 0
                for r in range(target_rank):
                    if np.any(U_ls2[:, r] != 0) or np.any(V_ls2[:, r] != 0) or np.any(W_ls2[:, r] != 0):
                        active += 1
                if active < best_rank:
                    best_U, best_V, best_W = U_ls2, V_ls2, W_ls2
                    best_rank = active
    
    # Round to half-integers and verify
    best_U = np.round(best_U * 2) / 2.0
    best_V = np.round(best_V * 2) / 2.0
    best_W = np.round(best_W * 2) / 2.0
    
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
