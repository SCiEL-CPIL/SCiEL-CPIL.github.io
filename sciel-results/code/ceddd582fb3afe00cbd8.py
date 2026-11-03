<<<<<<< SEARCH
def _lbfgs_search(n, m, p, rank, deadline, U_init, V_init, W_init):
    """
    L-BFGS-B gradient-based optimization exploiting bilinear structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    U_init = np.array(U_init, dtype=np.float64)
    V_init = np.array(V_init, dtype=np.float64)
    W_init = np.array(W_init, dtype=np.float64)
    
    max_restarts = 10
    for trial in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(trial * 100 + rank)
        
        # Initialize near the known solution with small perturbation
        if rank <= 55:
            U_c = U_init[:, :rank].copy() + 0.05 * np.random.randn(nm, rank)
            V_c = V_init[:, :rank].copy() + 0.05 * np.random.randn(mp, rank)
            W_c = W_init[:, :rank].copy() + 0.05 * np.random.randn(pn, rank)
        else:
            U_c = np.random.randn(nm, rank) * 0.5
            V_c = np.random.randn(mp, rank) * 0.5
            W_c = np.random.randn(pn, rank) * 0.5
        
        params = np.concatenate([U_c.flatten(), V_c.flatten(), W_c.flatten()])
        
        def objective_and_grad(params):
            if time.time() > deadline - 5:
                return float('inf'), np.zeros_like(params)
            
            U_p = params[:nm*rank].reshape(n, m, rank)
            V_p = params[nm*rank:nm*rank + mp*rank].reshape(m, p, rank)
            W_p = params[nm*rank + mp*rank:].reshape(p, n, rank)
            
            err = 0.0
            grad_U = np.zeros_like(U_p)
            grad_V = np.zeros_like(V_p)
            grad_W = np.zeros_like(W_p)
            
            for i in range(n):
                for k in range(p):
                    U_i = U_p[i]  # shape (m, rank)
                    V_k = V_p[:, k]  # shape (m, rank)
                    W_ki = W_p[k, i]  # shape (rank,)
                    
                    M_ik = (U_i * W_ki) @ V_k.T
                    diff = M_ik - np.eye(m)
                    err += np.sum(diff**2)
                    
                    # Gradient computation
                    dM_dU = np.diag(W_ki) @ V_k.T  # shape (m, m)
                    dM_dV = U_i @ np.diag(W_ki)  # shape (m, m)
                    
                    grad_U[i] = 2 * (diff @ dM_dU.T)  # shape (m, rank)
                    grad_V[:, k] = 2 * (dM_dV @ diff.T)  # shape (m, rank)
                    
                    for r in range(rank):
                        dM_dWr = np.outer(U_i[:, r], V_k[:, r])  # shape (m, m)
                        grad_W[k, i, r] += 2 * np.sum(diff * dM_dWr)
            
            return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res = minimize(objective_and_grad, params, jac=True, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < 1e-8:
                U_sol = res.x[:nm*rank].reshape(n*m, rank)
                V_sol = res.x[nm*rank:nm*rank + mp*rank].reshape(m*p, rank)
                W_sol = res.x[nm*rank + mp*rank:].reshape(p*n, rank)
                
                # Verify
                err = _check_validity_residual(U_sol, V_sol, W_sol, n, m, p, deadline)
                if err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    err_r = _check_validity_residual(U_r, V_r, W_r, n, m, p, deadline)
                    if err_r < 1e-10:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    else:
                        # Use unrounded if close enough
                        if err < 1e-10:
                            return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        except Exception:
            pass
    
    return None
=======
def _lbfgs_search(n, m, p, rank, deadline, U_init, V_init, W_init):
    """
    L-BFGS-B gradient-based optimization exploiting bilinear structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    U_init = np.asarray(U_init, dtype=np.float64)
    V_init = np.asarray(V_init, dtype=np.float64)
    W_init = np.asarray(W_init, dtype=np.float64)
    
    max_restarts = 8
    for trial in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(trial * 100 + rank)
        
        # Initialize near the known solution with small perturbation
        if rank <= 55:
            U_c = U_init[:, :rank].copy() + 0.03 * np.random.randn(nm, rank)
            V_c = V_init[:, :rank].copy() + 0.03 * np.random.randn(mp, rank)
            W_c = W_init[:, :rank].copy() + 0.03 * np.random.randn(pn, rank)
        else:
            U_c = np.random.randn(nm, rank) * 0.3
            V_c = np.random.randn(mp, rank) * 0.3
            W_c = np.random.randn(pn, rank) * 0.3
        
        params = np.concatenate([U_c.flatten(), V_c.flatten(), W_c.flatten()])
        
        def objective_and_grad(params):
            if time.time() > deadline - 10:
                return float('inf'), np.zeros_like(params)
            
            U_p = params[:nm*rank].reshape(n, m, rank)
            V_p = params[nm*rank:nm*rank + mp*rank].reshape(m, p, rank)
            W_p = params[nm*rank + mp*rank:].reshape(p, n, rank)
            
            err = 0.0
            grad_U = np.zeros_like(U_p)
            grad_V = np.zeros_like(V_p)
            grad_W = np.zeros_like(W_p)
            I_m = np.eye(m)
            
            for i in range(n):
                for k in range(p):
                    U_i = U_p[i]  # shape (m, rank)
                    V_k = V_p[:, k]  # shape (m, rank)
                    W_ki = W_p[k, i]  # shape (rank,)
                    
                    M_ik = (U_i * W_ki) @ V_k.T
                    diff = M_ik - I_m
                    err += np.sum(diff**2)
                    
                    # Gradient: d/dU[i,j,r] = 2 * sum_{j2} diff[j1,j2] * W[k,i,r] * V[j2,k,r]
                    # = 2 * (diff @ (W[k,i] * V_k.T)) [j1, r] ... wait
                    # M_ik[j1,j2] = sum_r U[i,j1,r] * W[k,i,r] * V[j2,k,r]
                    # dM_ik[j1,j2]/dU[i,j1,r] = W[k,i,r] * V[j2,k,r]
                    # d err / dU[i,j1,r] = 2 * sum_{j2} diff[j1,j2] * W[k,i,r] * V[j2,k,r]
                    # = 2 * W[k,i,r] * sum_{j2} diff[j1,j2] * V[j2,k,r]
                    # = 2 * W[k,i,r] * (diff @ V_k.T)[j1, r]... no
                    # sum_{j2} diff[j1,j2] * V[j2,k,r] = (diff @ V_k.T)[j1, r]
                    # So grad_U[i,j1,r] = 2 * W[k,i,r] * (diff @ V_k.T)[j1, r]
                    
                    dV = diff @ V_k.T  # shape (m, rank)
                    grad_U[i] += 2 * dV * W_ki  # broadcast W_ki over rows
                    
                    # dM_ik[j1,j2]/dV[j2,k,r] = U[i,j1,r] * W[k,i,r]
                    # d err / dV[j2,k,r] = 2 * sum_{j1} diff[j1,j2] * U[i,j1,r] * W[k,i,r]
                    # = 2 * W[k,i,r] * (diff.T @ U_i)[j2, r]
                    dU = diff.T @ U_i  # shape (m, rank)
                    grad_V[:, k] += 2 * dU * W_ki
                    
                    # dM_ik[j1,j2]/dW[k,i,r] = U[i,j1,r] * V[j2,k,r]
                    # d err / dW[k,i,r] = 2 * sum_{j1,j2} diff[j1,j2] * U[i,j1,r] * V[j2,k,r]
                    # = 2 * sum_r' ... no, for fixed r:
                    # = 2 * sum_{j1,j2} diff[j1,j2] * U[i,j1,r] * V[j2,k,r]
                    # = 2 * (U_i[:,r] * diff * V_k[:,r])... 
                    # = 2 * np.sum(diff * np.outer(U_i[:,r], V_k[:,r]))
                    # Vectorized: for each r, grad_W[k,i,r] = 2 * np.sum(diff * np.outer(U_i[:,r], V_k[:,r]))
                    # = 2 * np.einsum('ab,ar,br->r', diff, U_i, V_k)
                    grad_W[k, i, :] += 2 * np.einsum('ab,ar,br->r', diff, U_i, V_k)
            
            return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res = minimize(objective_and_grad, params, jac=True, method='L-BFGS-B', 
                          options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
            
            if res.fun < 1e-10:
                U_sol = res.x[:nm*rank].reshape(n*m, rank)
                V_sol = res.x[nm*rank:nm*rank + mp*rank].reshape(m*p, rank)
                W_sol = res.x[nm*rank + mp*rank:].reshape(p*n, rank)
                
                # Verify
                err = _check_validity_residual(U_sol, V_sol, W_sol, n, m, p, deadline)
                if err < 1e-10:
                    # Try to round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    err_r = _check_validity_residual(U_r, V_r, W_r, n, m, p, deadline)
                    if err_r < 1e-10:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    else:
                        return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        except Exception:
            pass
    
    return None
>>>>>>> REPLACE

