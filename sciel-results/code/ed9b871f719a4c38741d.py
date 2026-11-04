"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat_2d = T.reshape(nm, mp * pn)  # (nm, mp*pn)
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        max_err = np.max(np.abs(T_approx - T))
        return max_err < 1e-9, max_err

    def compute_error_and_grad(params, target_rank):
        """
        Compute error and gradient for L-BFGS-B.
        params: flattened array of [U, V, W]
        """
        size_U = nm * target_rank
        size_V = mp * target_rank
        size_W = pn * target_rank
        
        U = params[:size_U].reshape(nm, target_rank)
        V = params[size_U:size_U+size_V].reshape(mp, target_rank)
        W = params[size_U+size_V:].reshape(pn, target_rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T_approx - T
        error = np.sum(diff * diff)
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def try_rank_als(target_rank, n_restarts=5, max_iter=100):
        """
        Try to find a rank-target_rank decomposition using ALS.
        This is a structured approach that may converge faster for some ranks.
        """
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: T_flat = U @ M_U where M_U[r, bc] = V[b,r]*W[c,r]
                M_U = np.stack([np.outer(V[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                # M_U shape: (target_rank, mp*pn)
                # Solve U @ M_U = T_flat_2d
                try:
                    U_new = T_flat_2d @ np.linalg.pinv(M_U)
                except:
                    break
                U = U_new
                
                # Update V: For each b, T[:,b,:] (nm x pn) = sum_r V[b,r] * outer(U[:,r], W[:,r])
                # M_V[r, ac] = U[a,r]*W[c,r], shape (target_rank, nm*pn)
                M_V = np.stack([np.outer(U[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                # T_flat_v: (mp, nm*pn) where T_flat_v[b, ac] = T[a, b, c]
                T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new = T_flat_v @ np.linalg.pinv(M_V)
                except:
                    break
                V = V_new
                
                # Update W: For each c, T[:, :, c] (nm x mp) = sum_r W[c,r] * outer(U[:,r], V[:,r])
                M_W = np.stack([np.outer(U[:, r], V[:, r]).reshape(-1) for r in range(target_rank)])
                # T_flat_w: (pn, nm*mp) where T_flat_w[c, ab] = T[a, b, c]
                T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = T_flat_w @ np.linalg.pinv(M_W)
                except:
                    break
                W = W_new
                
                # Check convergence
                ok, err = check_decomposition(U, V, W)
                if ok:
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    return True
                    
            # Final check after ALS
            ok, err = check_decomposition(U, V, W)
            if ok:
                if target_rank < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                return True
                
        return False

    def try_rank_l_bfgs(target_rank, n_restarts=5):
        """
        Try to find a rank-target_rank decomposition using L-BFGS-B.
        """
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(
                    compute_error_and_grad, 
                    params0, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
                )
                
                final_params = res.x
                size_U = nm * target_rank
                size_V = mp * target_rank
                U_opt = final_params[:size_U].reshape(nm, target_rank)
                V_opt = final_params[size_U:size_U+size_V].reshape(mp, target_rank)
                W_opt = final_params[size_U+size_V:].reshape(pn, target_rank)
                
                ok, err = check_decomposition(U_opt, V_opt, W_opt)
                if ok:
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                        best_rank = target_rank
                    return True
                    
            except Exception:
                pass
                
        return False

    def try_rank(target_rank):
        """
        Try both ALS and L-BFGS-B for the given rank.
        """
        # Try ALS first (often faster for structured problems)
        if try_rank_als(target_rank, n_restarts=3, max_iter=50):
            return True
        # Then try L-BFGS-B
        if try_rank_l_bfgs(target_rank, n_restarts=3):
            return True
        return False

    # Try ranks from 111 down to a reasonable limit
    # We know 112 works. We want to find if we can go lower.
    for target_rank in range(111, 49, -1):
        if time.time() > deadline - 20:
            break
        try_rank(target_rank)
        # If we found a valid decomposition at this rank, continue to lower ranks
        # The loop naturally continues, so no special handling needed.

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
