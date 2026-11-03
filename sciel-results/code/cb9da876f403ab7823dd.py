import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, T, tol=1e-9):
        """Check if sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c]"""
        D = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(D, T, atol=tol)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Function to flatten U, V, W into a single vector
    def flatten(U, V, W):
        return np.concatenate((U.flatten(), V.flatten(), W.flatten()))
    
    # Function to unflatten vector into U, V, W
    def unflatten(x, r):
        size_u = n*m*r
        size_v = m*p*r
        size_w = p*n*r
        U = x[:size_u].reshape(n*m, r)
        V = x[size_u:size_u+size_v].reshape(m*p, r)
        W = x[size_u+size_v:].reshape(p*n, r)
        return U, V, W
    
    def objective(x, r):
        U, V, W = unflatten(x, r)
        D = np.einsum('at,bt,ct->abc', U, V, W)
        E = T - D
        return np.sum(E**2)
        
    def gradient(x, r):
        U, V, W = unflatten(x, r)
        # Gradient of sum (T - sum U V W)^2 with respect to U, V, W
        # Let D = sum U V W
        # dL/dU[a,t] = -2 * sum_{b,c} (T[a,b,c] - D[a,b,c]) * V[b,t] * W[c,t]
        # dL/dV[b,t] = -2 * sum_{a,c} (T[a,b,c] - D[a,b,c]) * U[a,t] * W[c,t]
        # dL/dW[c,t] = -2 * sum_{a,b} (T[a,b,c] - D[a,b,c]) * U[a,t] * V[b,t]
        
        D = np.einsum('at,bt,ct->abc', U, V, W)
        E = T - D
        
        # grad_U shape (n*m, r)
        # grad_U[a,t] = -2 * sum_{b,c} E[a,b,c] V[b,t] W[c,t]
        # We can compute this using einsum
        # Intermediate: sum_c E[a,b,c] W[c,t] -> (a,b,t)
        # Then sum_b (intermediate) V[b,t] -> (a,t)
        grad_U = -2.0 * np.einsum('abc,bt,ct->at', E, V, W)
        grad_V = -2.0 * np.einsum('abc,at,ct->bt', E, U, W)
        grad_W = -2.0 * np.einsum('abc,at,bt->ct', E, U, V)
        
        return flatten(grad_U, grad_V, grad_W)
    
    def try_rank(r, time_budget, max_restarts=10):
        """Try to find a rank-r decomposition using L-BFGS-B."""
        best_solution = None
        best_loss = float('inf')
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Random initialization
            # Scale initialization to avoid numerical issues initially
            scale = 1.0
            rng = np.random.default_rng(restart)
            x0 = rng.normal(0, scale, size=(n*m*r + m*p*r + p*n*r))
            
            # Set options for minimize
            options = {
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=lambda x: gradient(x, r),
                    args=(r,),
                    options=options
                )
            except Exception as e:
                continue
                
            if not res.success and res.fun > best_loss:
                # If it didn't converge but improved, keep it
                pass
                
            current_loss = res.fun
            if current_loss < best_loss:
                best_loss = current_loss
                U, V, W = unflatten(res.x, r)
                
                # Check if close to integer/half-integer grid
                # We try to round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded
                
                # If very close, maybe refine? 
                # For now, just store best float solution
                best_solution = (U, V, W)
                
        return best_solution
    
    # Try ranks from 25 to 35
    # 36 is schoolbook
    for r in range(25, 36):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Allocate time budget for this rank
        time_used = time.time() - start_time
        time_remaining = time_limit - time_used
        if time_remaining < 5:
            break
            
        # Give more time to lower ranks? Or equal time?
        # Let's give equal time slices for now, maybe 5-10s per rank
        time_budget_for_rank = min(10.0, time_remaining / (36 - r))
        
        result = try_rank(r, time_budget_for_rank)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
