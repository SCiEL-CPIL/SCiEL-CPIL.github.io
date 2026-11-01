import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, atol=1e-9):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def try_rank_with_structure(r, num_restarts=30):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        # Number of variables: r * (nm + mp + pn)
        size = r * (n*m + m*p + p*n)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Structured initialization: start with schoolbook pattern and perturb
            params = np.zeros(size)
            
            # Initialize with schoolbook-like structure
            U_init = np.zeros((n*m, r))
            V_init = np.zeros((m*p, r))
            W_init = np.zeros((p*n, r))
            
            # Use a mix of random and structured initialization
            if restart % 3 == 0:
                # Pure random
                params = np.random.randn(size) * 0.3
            elif restart % 3 == 1:
                # Structured: copy schoolbook and add noise
                for col in range(min(r, n*m*p)):
                    i = col // (m*p)
                    j = (col // p) % m
                    k = col % p
                    if i < n and j < m and k < p:
                        U_init[i*m+j, col] = 1.0
                        V_init[j*p+k, col] = 1.0
                        W_init[k*n+i, col] = 1.0
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.1
            else:
                # Sparse random
                params = np.random.randn(size)
                mask = np.random.rand(size) > 0.5
                params[mask] = 0
                params *= 0.5
            
            def objective(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                return compute_error(U, V, W)
            
            def gradient(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return grad
            
            # Use L-BFGS-B with gradient
            res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300})
            
            if time.time() > deadline:
                break
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
                if err < 1e-8:
                    U = res.x[:r*n*m].reshape(n*m, r)
                    V = res.x[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                    W = res.x[r*(n*m+m*p):].reshape(p*n, r)
                    
                    # Try rounding to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                    elif check_decomposition(U, V, W):
                        return U, V, W
        
        return None
    
    # Try ranks from low to high
    for rank in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        res = try_rank_with_structure(rank, num_restarts=20)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
