import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    
    # Build the target tensor
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    deadline = time.time() + 135
    
    def compute_error(U, V, W, T):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_approx - T)**2)
    
    def als_step(U, V, W, T, r):
        nm, mp, pn = n*m, m*p, p*n
        
        M_U = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
        M_V = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
        M_W = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
        
        for a in range(nm):
            target = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M_U.T, target, rcond=None)
            except:
                pass
        
        for b in range(mp):
            target = T[:, b, :].reshape(nm, pn).flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M_V.T, target, rcond=None)
            except:
                pass
        
        for c in range(pn):
            target = T[:, :, c].reshape(nm, mp).flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M_W.T, target, rcond=None)
            except:
                pass
        
        return U, V, W

    def polish(U, V, W, T, r):
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        err_int = compute_error(U_int, V_int, W_int, T)
        if err_int < 1e-10:
            return U_int, V_int, W_int
        
        # Try rounding to nearest half-integer
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        err_half = compute_error(U_half, V_half, W_half, T)
        if err_half < 1e-10:
            return U_half, V_half, W_half
            
        # Try a few local perturbations around the current solution before giving up on rounding
        # This is a heuristic to escape near-misses
        for _ in range(5):
            U_pert = U + np.random.normal(0, 0.01, U.shape)
            V_pert = V + np.random.normal(0, 0.01, V.shape)
            W_pert = W + np.random.normal(0, 0.01, W.shape)
            
            U_int = np.round(U_pert)
            V_int = np.round(V_pert)
            W_int = np.round(W_pert)
            if compute_error(U_int, V_int, W_int, T) < 1e-10:
                return U_int, V_int, W_int
                
            U_half = np.round(U_pert * 2) / 2
            V_half = np.round(V_pert * 2) / 2
            W_half = np.round(W_pert * 2) / 2
            if compute_error(U_half, V_half, W_half, T) < 1e-10:
                return U_half, V_half, W_half

        return U, V, W

    # Search from lower ranks
    # Lower bound for 2x3x9 is likely around 27 (9 * rank(2x3x1))
    # rank(2x3x1) is 3. So 27 is a strong candidate.
    # Schoolbook is 54.
    # Let's try a range.
    for r in range(27, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        trials = 10 if r < 35 else 5
        for trial in range(trials):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 1000 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            for col_idx in range(r):
                norm = np.sqrt(np.sum(U[:, col_idx]**2) + np.sum(V[:, col_idx]**2) + np.sum(W[:, col_idx]**2))
                if norm > 1e-10:
                    U[:, col_idx] /= norm
                    V[:, col_idx] /= norm
                    W[:, col_idx] /= norm
            
            success = False
            prev_error = compute_error(U, V, W, T)
            
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                    
                U, V, W = als_step(U, V, W, T, r)
                
                error = compute_error(U, V, W, T)
                
                if error < 1e-10:
                    success = True
                    break
                elif error > 1e6:
                    break
                elif abs(error - prev_error) < 1e-15:
                    if error < 1e-6:
                        success = True
                    break
                prev_error = error
            
            if success:
                U, V, W = polish(U, V, W, T, r)
                if compute_error(U, V, W, T) < 1e-10:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        
        if best_rank == r:
            break
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
