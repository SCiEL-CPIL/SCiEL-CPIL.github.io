<<<<<<< SEARCH
def _polish_rank31(U, V, W, n, m, p, T_true, time_budget):
    """High-precision polish pass for rank-31 decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    start_time = time.time()
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        
        # Use sum of squares for smooth optimization
        return np.sum(residual**2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
    except Exception:
        return U, V, W
    
    if result.fun < 1e-10:
        U_flat = result.x[:nm * rank]
        V_flat = result.x[nm * rank : nm * rank + mp * rank]
        W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U_polished = U_flat.reshape(nm, rank)
        V_polished = V_flat.reshape(mp, rank)
        W_polished = W_flat.reshape(pn, rank)
        return U_polished, V_polished, W_polished
    
    return U, V, W
=======
def _polish_rank31(U, V, W, n, m, p, T_true, time_budget):
    """High-precision polish pass for rank-31 decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    start_time = time.time()
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        
        # Use sum of squares for smooth optimization
        return np.sum(residual**2)
    
    def gradient(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        
        # Gradient: d/dU[i,r] sum_ijkr residual[i,j,k]^2
        # = 2 * sum_jk residual[i,j,k] * V[j,r] * W[k,r]
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', residual, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Use L-BFGS-B with analytic gradient for faster convergence
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 500})
    except Exception:
        # Fallback to numerical gradient
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
        except Exception:
            return U, V, W
    
    if result.fun < 1e-10:
        U_flat = result.x[:nm * rank]
        V_flat = result.x[nm * rank : nm * rank + mp * rank]
        W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U_polished = U_flat.reshape(nm, rank)
        V_polished = V_flat.reshape(mp, rank)
        W_polished = W_flat.reshape(pn, rank)
        return U_polished, V_polished, W_polished
    
    return U, V, W
>>>>>>> REPLACE

