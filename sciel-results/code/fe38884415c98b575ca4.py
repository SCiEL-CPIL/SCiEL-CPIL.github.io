import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    def objective_and_grad(params, R):
        # Unpack
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        # Compute tensor product sum
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Objective
        diff = M - T
        obj = np.sum(diff**2)
        
        # Gradient
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return obj, grad

    def try_rank(R, n_restarts=10):
        """Try to find a decomposition of rank R."""
        best_obj = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Using a smaller scale to avoid large gradients initially
            init_U = np.random.randn(nm, R) * 0.1
            init_V = np.random.randn(mp, R) * 0.1
            init_W = np.random.randn(pn, R) * 0.1
            
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            # Optimization
            try:
                res = opt.minimize(
                    objective_and_grad, 
                    params0, 
                    args=(R,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                
            if best_obj < 1e-10:
                break
                
        if best_obj < 1e-10 and best_params is not None:
            # Verify
            U = best_params[:nm*R].reshape(nm, R)
            V = best_params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = best_params[nm*R+mp*R:].reshape(pn, R)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(M, T, atol=1e-8):
                return U, V, W, best_obj
        return None, None, None, best_obj

    # Search from 83 down to 50
    # We want the lowest rank, so we iterate downwards.
    # If we find a solution at 83, we keep it, but try 82.
    
    current_best_rank = rank_sb
    current_best_U, current_best_V, current_best_W = U_sb, V_sb, W_sb
    
    # We prioritize finding *any* improvement.
    # Higher ranks are easier.
    # Let's try 83, 82, ...
    # If we find a solution, we update the best and continue.
    
    for R in range(rank_sb - 1, 49, -1):
        if time.time() > deadline - 10:
            break
            
        # Adjust restarts: more restarts for lower ranks?
        # Lower ranks are harder, so they need more exploration.
        # But if we run out of time, we can't finish.
        # Let's use a fixed 15 restarts.
        U, V, W, obj = try_rank(R, n_restarts=15)
        
        if U is not None:
            current_best_rank = R
            current_best_U, current_best_V, current_best_W = U, V, W
            # Don't break, try lower rank!
            
    # Return the best result found
    return {"U": current_best_U.tolist(), "V": current_best_V.tolist(), "W": current_best_W.tolist()}
