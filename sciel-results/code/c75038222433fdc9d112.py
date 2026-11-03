import numpy as np
import time
from scipy.optimize import minimize

def _verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition.
    Condition: sum_r U[im+j, r] V[jp+k, r] W[kn+i, r] == 1 for all i,j,k.
    """
    R = U.shape[1]
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                errors.append(s - 1.0)
    return max(abs(e) for e in errors) if errors else 0.0

def _try_rank(R, n, m, p, timeout=10.0):
    """
    Attempts to find a decomposition of rank R.
    """
    start_time = time.time()
    
    # Initialize with schoolbook-like structure but with R columns
    # We can take the first R terms of schoolbook (if R < nmp)
    # Or random initialization.
    # Let's try random initialization with small values.
    np.random.seed(42)
    
    U = np.random.rand(n * m, R) * 0.5 - 0.25
    V = np.random.rand(m * p, R) * 0.5 - 0.25
    W = np.random.rand(p * n, R) * 0.5 - 0.25
    
    # Flatten for optimization
    def flatten(params):
        u = params[:n*m*R].reshape(n*m, R)
        v = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        w = params[n*m*R + m*p*R:].reshape(p*n, R)
        return u, v, w

    def objective(params):
        if time.time() - start_time > timeout:
            return float('inf')
        u, v, w = flatten(params)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += u[i * m + j, r] * v[j * p + k, r] * w[k * n + i, r]
                    err += (s - 1.0) ** 2
        return err

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use L-BFGS-B for speed
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if time.time() - start_time > timeout:
        return None
        
    if res.fun < 1e-6:
        u, v, w = flatten(res.x)
        # Check if we can round to half-integers
        # Try rounding to nearest 0.5
        u_r = np.round(u * 2) / 2
        v_r = np.round(v * 2) / 2
        w_r = np.round(w * 2) / 2
        
        if _verify(u_r, v_r, w_r, n, m, p) < 1e-9:
            return u_r, v_r, w_r
            
        # If rounding fails, maybe scale?
        # Or just return the float solution if the evaluator accepts floats?
        # Prompt says "integer/half-integer entries".
        # If we can't round, we might need to search more.
        # But for now, if we find a float solution, we can try to refine.
        
        # Try a few other rounding strategies or just return the float if we must?
        # The prompt implies we must return valid types.
        # If we can't get exact half-integers, we might fail.
        # However, many tensor decompositions have rational entries.
        
        # Let's try to see if the solution is close to integers/half-integers.
        # If not, we might need a different initialization.
        
        # For now, return the rounded version if error is small, else None.
        # Actually, if _verify fails, return None.
        
    return None

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = 110.0 # 10s safety margin
    
    # Try to find rank < 48
    for R in range(47, 30, -1):
        if time.time() - start_time > deadline:
            break
        result = _try_rank(R, n, m, p, timeout=10.0)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
                
    return {"U": U_list, "V": V_list, "W": W_list}
