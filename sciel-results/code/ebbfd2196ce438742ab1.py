"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Fallback: schoolbook decomposition of rank 24
    U_fb = np.zeros((n*m, n*m*p), dtype=float)
    V_fb = np.zeros((m*p, n*m*p), dtype=float)
    W_fb = np.zeros((p*n, n*m*p), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    from scipy.optimize import minimize
    
    # Try to find lower-rank exact decompositions
    for rank in range(14, n*m*p):
        if time.time() > deadline - 15:
            break
        
        found = False
        n_tries = 100
        
        for trial in range(n_tries):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            def objective(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                            err += (val - 1.0) ** 2
                return err
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, params0, method='L-BFGS-B', 
                                 options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
            except Exception:
                continue
            
            if result.fun < 1e-10:
                # Extract and round to half-integers
                U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                # Verify exactness after rounding
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    best_rank = rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # Convert to list format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
