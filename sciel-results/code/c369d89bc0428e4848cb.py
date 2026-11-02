import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[u, v, w] = 1 if (u,v,w) corresponds to (i,j,k)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Schoolbook fallback
    rank_school = nm * mp * pn
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1.0
                V_school[v, col] = 1.0
                W_school[w, col] = 1.0
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # We suspect rank is around 20-30.
    # Let's try ranks from 20 to 35.
    ranks_to_try = list(range(20, 36))
    
    def verify(U, V, W):
        # Compute tensor from factors
        # T_comp = sum_r U[:,r] V[:,r] W[:,r]
        # Use einsum for speed
        T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_comp, T, atol=1e-9, rtol=1e-9)
        
    def round_and_verify(U, V, W):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
        
    for R in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        found = False
        # Number of restarts depends on time left
        # Estimate time per restart: ~0.1s to 1s
        # Let's do a fixed number of restarts, e.g., 50, but break if time is tight
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initialization
            # Scale to 1.0 to help convergence?
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_ = params[:nm*R].reshape(nm, R)
                V_ = params[nm*R:nm*R+mp*R].reshape(mp, R)
                W_ = params[nm*R+mp*R:].reshape(pn, R)
                
                T_comp = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
                diff = T_comp - T
                return np.sum(diff * diff)
                
            # Optimization
            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                U_opt = res.x[:nm*R].reshape(nm, R)
                V_opt = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W_opt = res.x[nm*R+mp*R:].reshape(pn, R)
                
                # Try to round
                res_rounded = round_and_verify(U_opt, V_opt, W_opt)
                if res_rounded is not None:
                    U_r, V_r, W_r = res_rounded
                    best_rank = R
                    best_solution = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    found = True
                    break # Found valid rank R, move to next lower R? No, we iterate ascending.
                    # If we found R=20, it's the best possible in this list.
                    # We should break the outer loop too?
                    # Yes, if we found a solution for R, and we are iterating 20, 21, 22...
                    # Then 20 is the minimum found.
                    # We can return immediately.
                    return best_solution
        
        # If we didn't find a solution for this R, continue to next R
        # If we found a solution, we already returned.
        
    return best_solution
