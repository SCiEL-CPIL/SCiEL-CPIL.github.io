import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_solution(U, V, W, rank):
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        return valid

    def objective_and_grad(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
        W = x[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
        
        err = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    diff = s - 1.0
                    err += diff * diff
                    for c in range(rank):
                        grad_U[i*m + j, c] += 2 * diff * V[j*p + k, c] * W[k*n + i, c]
                        grad_V[j*p + k, c] += 2 * diff * U[i*m + j, c] * W[k*n + i, c]
                        grad_W[k*n + i, c] += 2 * diff * U[i*m + j, c] * V[j*p + k, c]
        
        g = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return err, g

    # Try to find a better decomposition
    # Search ranks from 46 to 53
    for rank in range(46, 54):
        if time.time() > deadline:
            break
            
        found = False
        # A few restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            x0 = np.random.uniform(-1, 1, 3 * n * m * rank)
            
            # Use scipy minimize
            # We need a function that returns (f, g)
            # scipy minimize expects f and optionally jac
            # We can use a wrapper
            
            def fun(x):
                f, g = objective_and_grad(x, rank)
                return f, g
            
            res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                x_opt = res.x
                U = x_opt[:n*m*rank].reshape(n*m, rank)
                V = x_opt[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W = x_opt[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify_solution(U_r, V_r, W_r, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
