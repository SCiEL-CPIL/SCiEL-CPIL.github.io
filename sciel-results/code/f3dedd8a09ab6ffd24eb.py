import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    def check_decomposition(U, V, W, atol=1e-9):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # T_approx = U @ V.T @ W.T ? No, einsum is better.
        # T_approx[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def try_rank_with_structure(r, num_restarts=20):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5: # Leave 5s for final checks
                break
                
            # Initialization
            if restart == 0:
                # Start with schoolbook if r >= 24? No, r < 24 here.
                # Use structured initialization: project schoolbook onto rank r?
                # Just use random with small noise around a sparse structure.
                params = np.random.randn(size) * 0.5
            elif restart % 2 == 0:
                # Random
                params = np.random.randn(size) * 0.3
            else:
                # Sparse random
                params = np.random.randn(size)
                mask = np.random.rand(size) > 0.7
                params[mask] = 0
                params *= 0.5
            
            def objective(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                return compute_error(U, V, W)
            
            def gradient(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return grad
            
            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500})
                
                err = res.fun
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
                    
                    if err < 1e-8:
                        U = res.x[:r*nm].reshape(nm, r)
                        V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                        W = res.x[r*(nm+mp):].reshape(pn, r)
                        
                        # Rounding to 0.5
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_decomposition(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                        elif check_decomposition(U, V, W):
                            return U, V, W
            except Exception:
                pass
        
        return None

    # Try ranks from low to high
    # Rank 24 is the schoolbook. We try 20-23.
    # If the rank is 24, these will fail.
    rank_configs = [
        (23, 15),
        (22, 10),
        (21, 5),
        (20, 5),
    ]
    
    for rank, restarts in rank_configs:
        if time.time() > deadline - 10:
            break
        res = try_rank_with_structure(rank, num_restarts=restarts)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Verify fallback
    if not check_decomposition(U, V, W):
        # This should never happen, but return zeros if it does? 
        # No, better to return the matrices.
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
