import numpy as np
from scipy.optimize import minimize

def construct():
    import time
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[i, j, k] = 1 if C[i, k] = sum_j A[i, j] B[j, k]
    # Indices: U index i*m+j, V index j*p+k, W index k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Try to find a decomposition of rank 23
    # If found, return it.
    # Else return rank 26.
    
    def try_rank(r):
        if time.time() > deadline - 10:
            return None
            
        best_res = float('inf')
        best_params = None
        
        # Random restarts
        max_restarts = 20 # Adjust based on time
        
        for _ in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Scale to help optimization
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                if time.time() > deadline - 5:
                    return 1e10
                U_flat = params[:n*m*r]
                V_flat = params[n*m*r:n*m*r + m*p*r]
                W_flat = params[n*m*r + m*p*r:]
                
                U_mat = U_flat.reshape(n*m, r)
                V_mat = V_flat.reshape(m*p, r)
                W_mat = W_flat.reshape(p*n, r)
                
                # Compute tensor sum
                # T_sum[u, v, w] = sum_k U[u, k] V[v, k] W[w, k]
                # This is slow to compute in loop.
                # Use einsum
                T_sum = np.einsum('uk,vk,wk->uvw', U_mat, V_mat, W_mat)
                
                # Residual
                res = T_sum - T
                return np.sum(res**2)
            
            # Use Nelder-Mead or L-BFGS-B
            # L-BFGS-B might be better for large r
            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200})
                if res.fun < best_res:
                    best_res = res.fun
                    best_params = res.x
            except Exception:
                pass
            
            if best_res < 1e-6:
                break
        
        if best_params is not None and best_res < 1e-9:
            U_mat = best_params[:n*m*r].reshape(n*m, r)
            V_mat = best_params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_mat = best_params[n*m*r + m*p*r:].reshape(p*n, r)
            
            # Convert to lists
            U_list = U_mat.tolist()
            V_list = V_mat.tolist()
            W_list = W_mat.tolist()
            
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try rank 23
    result = try_rank(23)
    if result:
        return result
        
    # Fallback to rank 26 (current implementation)
    # ... (insert current rank 26 code) ...
    
    return {"U": U, "V": V, "W": W}
