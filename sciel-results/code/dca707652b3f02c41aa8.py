import numpy as np
import time
import sys
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    # Schoolbook decomposition as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 140.0  # 140s search, 10s safety margin
    
    # Precompute target tensor
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0
    
    def try_rank(r):
        """Try to find a decomposition of rank r using L-BFGS-B optimization."""
        if time.time() > deadline:
            return None
        
        # Objective: minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
        # Using einsum for efficiency
        def objective_and_grad(params):
            if time.time() > deadline:
                return np.inf, np.zeros_like(params)
            
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - TARGET_TENSOR
            obj = np.sum(diff**2)
            
            # Gradient computation
            # dT/dc = sum_r U[a,r]*V[b,r]*W[c,r]
            # d_obj/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            # d_obj/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # d_obj/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
            
            # grad_W[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            # = 2 * einsum('abc,ar,br->cr', diff, U, V)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            
            # grad_V[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return obj, grad
        
        # Try several random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + r * 100 + attempt)
            scale = 0.5
            U0 = np.random.randn(n*m, r) * scale
            V0 = np.random.randn(m*p, r) * scale
            W0 = np.random.randn(p*n, r) * scale
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.fun < 1e-6:
                    # Extract solution
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Try to recover rational values
                    rational_solution = recover_rational(U, V, W, r, n, m, p, TARGET_TENSOR)
                    if rational_solution is not None:
                        return rational_solution
                        
            except Exception as e:
                continue
        
        return None
    
    def recover_rational(U, V, W, r, n, m, p, T_target):
        """Try to recover integer/half-integer values from float solution."""
        if time.time() > deadline:
            return None
        
        # Scale each matrix to try to get rational values
        # First, try to find a common scaling factor
        all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        nonzero = all_vals[np.abs(all_vals) > 1e-10]
        
        if len(nonzero) == 0:
            return None
        
        # Try different scaling factors
        for scale_factor in [1, 2, 4, 8, 16, 32]:
            U_s = U * scale_factor
            V_s = V * scale_factor
            W_s = W * scale_factor
            
            # Round to nearest half-integer
            U_r = np.rint(U_s * 2) / 2
            V_r = np.rint(V_s * 2) / 2
            W_r = np.rint(W_s * 2) / 2
            
            # Verify
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_target, atol=1e-8):
                # Verify all entries are integer or half-integer
                for mat in [U_r, V_r, W_r]:
                    doubled = mat * 2
                    if not np.allclose(doubled, np.rint(doubled), atol=1e-10):
                        break
                else:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        # Also try without scaling
        U_r = np.rint(U * 2) / 2
        V_r = np.rint(V * 2) / 2
        W_r = np.rint(W * 2) / 2
        
        T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_check, T_target, atol=1e-8):
            for mat in [U_r, V_r, W_r]:
                doubled = mat * 2
                if not np.allclose(doubled, np.rint(doubled), atol=1e-10):
                    break
            else:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        return None
    
    # Try ranks from 35 down to 30
    for r in range(35, 29, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
