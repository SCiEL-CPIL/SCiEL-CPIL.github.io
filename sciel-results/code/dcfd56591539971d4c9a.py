import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find rank 16
    target_rank = 16
    start_time = time.time()
    deadline = start_time + 120 # 120s limit, keep margin
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        np.random.seed(attempt)
        # Small integers to help convergence
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        W = np.random.randint(-1, 2, size=(p*n, target_rank))
        
        # Define the residual function
        def residual(flat_params):
            U_r = flat_params[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = flat_params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_r = flat_params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            res = np.zeros((n, m, m, p, n, p))
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for ip in range(n):
                                for kp in range(p):
                                    # Sum over c of W[kp*n+ip, c] * U[i*m+j, c] * V[jp*p+kp, c]
                                    val = np.sum(W_r[kp*n+ip, :] * U_r[i*m+j, :] * V_r[jp*p+kp, :])
                                    target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                    res[i, j, jp, k, ip, kp] = val - target
            return res.flatten()
            
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            from scipy.optimize import least_squares
            result = least_squares(residual, x0, method='lm', max_nfev=1000)
            if result.success or np.linalg.norm(result.fun) < 1e-6:
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Round to half-integers
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify
                if verify(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        except Exception:
            continue
            
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def verify(U, V, W, n, m, p):
    # U: (nm, r), V: (mp, r), W: (pn, r)
    r = U.shape[1]
    # Check if U, V, W are half-integers
    if not np.allclose(U * 2, np.round(U * 2)): return False
    if not np.allclose(V * 2, np.round(V * 2)): return False
    if not np.allclose(W * 2, np.round(W * 2)): return False
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            val = np.sum(W[kp*n+ip, :] * U[i*m+j, :] * V[jp*p+kp, :])
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            if not np.isclose(val, target, atol=1e-9):
                                return False
    return True
