import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 8, 20, 10

    # Build target tensor T with shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, tol=1e-9):
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def check_half_int(M, tol=1e-6):
        for val in M.flatten():
            if abs(2*val - round(2*val)) > tol:
                return False
        return True

    def round_half_int(M):
        return np.round(2*M) / 2.0

    def get_schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    def try_rank_als(r, max_iter=200, seed=0, tol=1e-8):
        """ALS-style iterative tensor decomposition with rounding check."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random + structure-aware init
        # Use block structure: for each (i,k) pair, there are m=4 terms
        # Try structured init: group by (i,k)
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        # Normalize
        for _ in range(3):
            nU = np.linalg.norm(U)
            nV = np.linalg.norm(V)
            nW = np.linalg.norm(W)
            U /= (nU + 1e-10)
            V /= (nV + 1e-10)
            W /= (nW + 1e-10)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Update U: T[i, (j,k)] = sum_c U[i,c] * V[j,c] * W[k,c]
            # M_u[c, (j,k)] = V[j,c] * W[k,c]
            M_u = (V[:, :, None] * W[None, :, :]).reshape(r, -1)  # (r, mp*pn)
            T_flat = T.reshape(nm, -1)  # (nm, mp*pn)
            # Solve: M_u.T @ U.T = T_flat.T
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_u.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V
            M_v = (U[:, :, None] * W[None, :, :]).reshape(r, -1)  # (r, nm*pn)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W
            M_w = (U[:, :, None] * V[None, :, :]).reshape(r, -1)  # (r, nm*mp)
            T_flat_w = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Calculate error
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < tol:
                    # Try rounding
                    U_r = round_half_int(U)
                    V_r = round_half_int(V)
                    W_r = round_half_int(W)
                    if verify(U_r, V_r, W_r, tol=1e-5):
                        return U_r, V_r, W_r
                    # Try integer rounding
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if verify(U_i, V_i, W_i, tol=1e-5):
                        return U_i, V_i, W_i
        
        # Final check on best found
        U_r = round_half_int(best_U)
        V_r = round_half_int(best_V)
        W_r = round_half_int(best_W)
        if verify(U_r, V_r, W_r, tol=1e-5):
            return U_r, V_r, W_r
        return None

    def try_rank_structured(r, max_iter=200, seed=0):
        """Try with structured initialization based on block form."""
        rng = np.random.RandomState(seed)
        
        # Structured init: for each (i,k), have m terms
        # Total terms needed: n*p*m = 40, but we have r terms
        # Distribute r terms across the (i,k) groups
        n_groups = n * p  # 10 groups of (i,k)
        terms_per_group = r // n_groups
        extra = r % n_groups
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        col = 0
        for i in range(n):
            for k in range(p):
                n_terms = terms_per_group + (1 if (i * p + k) < extra else 0)
                for t in range(n_terms):
                    if col >= r:
                        break
                    # For this (i,k) group, distribute over j=0..m-1
                    j = t % m
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        
        # Add random perturbation for remaining columns
        if col < r:
            for c in range(col, r):
                U[:, c] = rng.randn(nm) * 0.1
                V[:, c] = rng.randn(mp) * 0.1
                W[:, c] = rng.randn(pn) * 0.1
        
        # Now run ALS from this init
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            M_u = (V[:, :, None] * W[None, :, :]).reshape(r, -1)
            T_flat = T.reshape(nm, -1)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_u.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            M_v = (U[:, :, None] * W[None, :, :]).reshape(r, -1)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            M_w = (U[:, :, None] * V[None, :, :]).reshape(r, -1)
            T_flat_w = T.transpose(2, 0, 1).reshape(pn, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-8:
                    U_r = round_half_int(U)
                    V_r = round_half_int(V)
                    W_r = round_half_int(W)
                    if verify(U_r, V_r, W_r, tol=1e-5):
                        return U_r, V_r, W_r
        
        U_r = round_half_int(best_U)
        V_r = round_half_int(best_V)
        W_r = round_half_int(best_W)
        if verify(U_r, V_r, W_r, tol=1e-5):
            return U_r, V_r, W_r
        return None

    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Schoolbook fallback
    U_s, V_s, W_s = get_schoolbook()
    best_result = (U_s, V_s, W_s)
    best_rank = n * m * p

    # Try ranks from 39 down to 25
    # Prioritize structured init for higher ranks, random for lower
    ranks_to_try = list(range(39, 24, -1))

    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 8 if r <= 35 else 5
        
        for seed in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Try structured first for higher ranks
            if r >= 35 and seed < n_restarts // 2:
                res = try_rank_structured(r, max_iter=150, seed=seed)
            else:
                res = try_rank_als(r, max_iter=150, seed=seed)
            
            if res is not None:
                U, V, W = res
                if verify(U, V, W, tol=1e-6) and check_half_int(U) and check_half_int(V) and check_half_int(W):
                    best_rank = r
                    best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            continue
        else:
            pass

    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
