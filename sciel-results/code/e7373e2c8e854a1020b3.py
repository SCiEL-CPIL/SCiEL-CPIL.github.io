import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    deadline = time.time() + 105
    n, m, p = DIMENSIONS
    
    # Build target tensor T[(i,j),(j,k),(k,i)] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook initialization
    school_rank = n * m * p
    U_sb = np.zeros((n*m, school_rank))
    V_sb = np.zeros((m*p, school_rank))
    W_sb = np.zeros((p*n, school_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = school_rank
    
    def compute_error(U, V, W):
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_decomp - target)**2)
    
    def als_step(U, V, W, max_iter=60, tol=1e-12):
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        prev_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            M_U = np.einsum('br,cr->rbc', V, W).reshape(r, m*p * p*n)
            target_flat = target.reshape(n*m, m*p * p*n)
            U = np.linalg.lstsq(M_U, target_flat.T, rcond=None)[0].T
            
            # Update V
            M_V = np.einsum('ar,cr->rac', U, W).reshape(r, n*m * p*n)
            target_flat_v = target.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
            V = np.linalg.lstsq(M_V, target_flat_v.T, rcond=None)[0].T
            
            # Update W
            M_W = np.einsum('ar,br->rab', U, V).reshape(r, n*m * m*p)
            target_flat_w = target.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            W = np.linalg.lstsq(M_W, target_flat_w.T, rcond=None)[0].T
            
            err = compute_error(U, V, W)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err
    
    # Try to reduce rank progressively
    for target_rank in range(school_rank - 1, max(40, school_rank - 60), -1):
        if time.time() > deadline - 20:
            break
        
        if target_rank <= best_rank:
            init_U = best_U[:, :target_rank]
            init_V = best_V[:, :target_rank]
            init_W = best_W[:, :target_rank]
        else:
            init_U = np.random.randn(n*m, target_rank)
            init_V = np.random.randn(m*p, target_rank)
            init_W = np.random.randn(p*n, target_rank)
        
        U_new, V_new, W_new, err = als_step(init_U, init_V, init_W)
        
        if err < 1e-6:
            # Try rounding to half-integers
            U_r = np.round(U_new * 2) / 2
            V_r = np.round(V_new * 2) / 2
            W_r = np.round(W_new * 2) / 2
            err_r = compute_error(U_r, V_r, W_r)
            if err_r < 1e-6:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
            else:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
    
    # Final rounding
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    err_r = compute_error(U_r, V_r, W_r)
    if err_r < 1e-6:
        best_U, best_V, best_W = U_r, V_r, W_r
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
