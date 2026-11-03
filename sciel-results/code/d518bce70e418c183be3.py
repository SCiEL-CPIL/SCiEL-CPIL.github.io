import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def try_rank(r, max_attempts=100):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                return False
            
            np.random.seed(hash((r, attempt)) % (2**32))
            
            # Try different initializations
            scale = [0.5, 1.0, 2.0, -1.0]
            s = scale[attempt % len(scale)]
            
            U_init = np.random.randn(n*m, r) * s
            V_init = np.random.randn(m*p, r) * s
            W_init = np.random.randn(p*n, r) * s
            
            def objective(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_p = params[n*m*r+m*p*r:].reshape(p*n, r)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            val = 0.0
                            for c in range(r):
                                val += U_p[i*m+j, c] * V_p[j*p+k, c] * W_p[k*n+i, c]
                            err += (val - 1.0)**2
                return err
            
            def gradient(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_p = params[n*m*r+m*p*r:].reshape(p*n, r)
                
                grad_U = np.zeros_like(U_p)
                grad_V = np.zeros_like(V_p)
                grad_W = np.zeros_like(W_p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_p[i*m+j, c] * V_p[j*p+k, c] * W_p[k*n+i, c]
                            err = 2.0 * (val - 1.0)
                            for c in range(r):
                                grad_U[i*m+j, c] += err * V_p[j*p+k, c] * W_p[k*n+i, c]
                                grad_V[j*p+k, c] += err * U_p[i*m+j, c] * W_p[k*n+i, c]
                                grad_W[k*n+i, c] += err * U_p[i*m+j, c] * V_p[j*p+k, c]
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            except Exception:
                continue
            
            if result.fun < 1e-10:
                U_p = result.x[:n*m*r].reshape(n*m, r)
                V_p = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_p = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Try to round to half-integers
                U_round = np.round(U_p * 2) / 2
                V_round = np.round(V_p * 2) / 2
                W_round = np.round(W_p * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_round[i*m+j, c] * V_round[j*p+k, c] * W_round[k*n+i, c]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = r
                    best_U, best_V, best_W = U_round, V_round, W_round
                    return True
        
        return False
    
    # Try ranks from low to high
    for r in range(1, best_rank):
        if time.time() > deadline - 30:
            break
        if try_rank(r):
            # Found a valid decomposition at rank r, try lower
            continue
        else:
            # If we couldn't find one at rank r, it might still exist
            # But for now, assume schoolbook is best if we can't find lower
            pass
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
