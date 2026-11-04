import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the multiplication tensor using vectorized einsum."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def objective_and_grad(params, rank):
        """Objective function and gradient for scipy optimization."""
        U = params[:n*m*rank].reshape(n, m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        residual = T - np.ones((n, m, p))
        
        obj = np.sum(residual ** 2)
        
        grad_U = 2 * np.einsum('ijk,jkr,kir->ijr', residual, V, W)
        grad_V = 2 * np.einsum('ijk,ijr,kir->jkr', residual, U, W)
        grad_W = 2 * np.einsum('ijk,ijr,jkr->kir', residual, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return obj, grad

    # Trivial (schoolbook) decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1

    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()

    np.random.seed(42)
    
    for target_rank in range(trivial_rank - 1, max(70, trivial_rank - 35), -1):
        if time.time() > deadline - 3:
            break
        
        found = False
        num_trials = 10
        
        for trial in range(num_trials):
            if time.time() > deadline - 3:
                break
            
            # Initialize near the trivial solution by projecting
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(
                    lambda x: objective_and_grad(x, target_rank)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, target_rank)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 5000, 'gtol': 1e-7}
                )
                
                if time.time() > deadline - 3:
                    break
                
                params_opt = result.x
                U_opt = params_opt[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = params_opt[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = params_opt[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                if check_decomposition(U_opt, V_opt, W_opt):
                    best_rank = target_rank
                    best_U = U_opt.copy()
                    best_V = V_opt.copy()
                    best_W = W_opt.copy()
                    found = True
                    break
                    
            except Exception:
                continue
        
        if found:
            break
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
