import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Precompute target tensor
    # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def check_decomposition(U, V, W, T_target):
        """Check if U, V, W form a valid exact decomposition with half-integer entries."""
        # Check if entries are close to half-integers
        U_check = U * 2
        V_check = V * 2
        W_check = W * 2
        
        U_int = np.rint(U_check)
        V_int = np.rint(V_check)
        W_int = np.rint(W_check)
        
        err_U = np.max(np.abs(U_check - U_int))
        err_V = np.max(np.abs(V_check - V_int))
        err_W = np.max(np.abs(W_check - W_int))
        
        if err_U < 1e-6 and err_V < 1e-6 and err_W < 1e-6:
            U_r = U_int / 2.0
            V_r = V_int / 2.0
            W_r = W_int / 2.0
            
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_target, atol=1e-8):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        return None

    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
            
        # Objective function
        def objective_and_grad(params):
            if time.time() > deadline:
                return np.inf, np.zeros_like(params)
            
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - TARGET_TENSOR
            obj = np.sum(diff**2)
            
            # Gradients
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return obj, grad
        
        # Strategy: Try sparse integer initializations
        # Focus on rank 25 and 24
        num_attempts = 100 if r <= 25 else 50
        found = None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + r * 100 + attempt)
            
            # Sparse initialization
            U0 = np.zeros((n*m, r))
            V0 = np.zeros((m*p, r))
            W0 = np.zeros((p*n, r))
            
            # Place random +/-1 entries
            # Density tuned for rank 25
            density = 0.5 
            for _ in range(int(n*m*r*density)):
                i = np.random.randint(n*m)
                j = np.random.randint(r)
                U0[i, j] = np.random.choice([-1, 1])
                
            for _ in range(int(m*p*r*density)):
                i = np.random.randint(m*p)
                j = np.random.randint(r)
                V0[i, j] = np.random.choice([-1, 1])
                
            for _ in range(int(p*n*r*density)):
                i = np.random.randint(p*n)
                j = np.random.randint(r)
                W0[i, j] = np.random.choice([-1, 1])
            
            # Add small noise
            U0 += np.random.randn(n*m, r) * 0.01
            V0 += np.random.randn(m*p, r) * 0.01
            W0 += np.random.randn(p*n, r) * 0.01
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 100}
                )
                
                if res.fun < 1e-8:
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    solution = check_decomposition(U, V, W, TARGET_TENSOR)
                    if solution is not None:
                        return solution
                        
            except Exception:
                continue
        
        return None

    # Schoolbook fallback
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

    # Search ranks from low to high
    # Prioritize 25 and 24
    for r in [25, 24, 26, 27, 28]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result

    # If optimization fails, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
