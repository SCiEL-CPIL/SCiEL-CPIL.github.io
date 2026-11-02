import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # T[i, j, l] = sum_r U[i*m + l, r] * V[l*p + j, r] * W[j*n + i, r]
    # Target T[i, j, l] = 1 for all i, j, l
    
    acc = np.zeros((n, p, m))
    for r_idx in range(r):
        u_2d = U[:, r_idx].reshape(n, m)
        v_2d = V[:, r_idx].reshape(m, p)
        w_2d = W[:, r_idx].reshape(p, n)
        for i in range(n):
            for j in range(p):
                for l in range(m):
                    acc[i, j, l] += u_2d[i, l] * v_2d[l, j] * w_2d[j, i]
    
    target = np.ones((n, p, m))
    return np.allclose(acc, target, atol=1e-9)

def _reduce_rank_iterative(U, V, W, n, m, p, deadline):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    while time.time() < deadline - 5:
        r = U.shape[1]
        if r == 0: break
        
        # Build column vectors representing the tensor contributions
        col_vecs = np.zeros((r, n * p * m), dtype=float)
        for c in range(r):
            u_2d = U[:, c].reshape(n, m)
            v_2d = V[:, c].reshape(m, p)
            w_2d = W[:, c].reshape(p, n)
            
            t = np.zeros((n, p, m))
            for i in range(n):
                for j in range(p):
                    for l in range(m):
                        t[i, j, l] = u_2d[i, l] * v_2d[l, j] * w_2d[j, i]
            col_vecs[c] = t.flatten()
            
        # Check for zero columns
        zero_cols = [c for c in range(r) if np.allclose(col_vecs[c], 0, atol=1e-12)]
        if zero_cols:
            keep = [c for c in range(r) if c not in zero_cols]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            continue
        
        # Find null space
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(col_vecs, full_matrices=False)
        except np.linalg.LinAlgError:
            break
            
        tol = 1e-10 * (max(s_vals) if len(s_vals) > 0 else 1.0)
        rank_cv = sum(1 for s in s_vals if s > tol)
        
        if rank_cv < r:
            null_vecs = Vt_svd[rank_cv:]
            removed = False
            for nv in null_vecs:
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) < 1e-12: continue
                
                c = max_idx
                v_c = nv[c]
                
                keep = [k for k in range(r) if k != c]
                scales = np.ones(len(keep))
                for idx, k in enumerate(keep):
                    beta_k = -nv[k] / v_c
                    factor = 1 + beta_k
                    if abs(factor) < 1e-12:
                         scales[idx] = 0.0
                    else:
                         s = np.cbrt(factor)
                         scales[idx] = s
                 
                U_new = U[:, keep].copy()
                V_new = V[:, keep].copy()
                W_new = W[:, keep].copy()
                
                for idx in range(len(keep)):
                    U_new[:, idx] *= scales[idx]
                    V_new[:, idx] *= scales[idx]
                    W_new[:, idx] *= scales[idx]
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    removed = True
                    break
            if removed:
                continue
        else:
            break
            
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 120
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(p):
            for l in range(m):
                U[i * m + l, r] = 1.0
                V[l * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                r += 1
                
    try:
        U, V, W = _reduce_rank_iterative(U, V, W, n, m, p, deadline)
    except Exception:
        pass
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
