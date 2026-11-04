import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=1e-9, rtol=1e-9)

    def error_func(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        return np.sum(diff**2)

    def solve_rank(R, max_restarts=200):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Better initialization: Random normal, scaled
            params0 = np.random.randn(total_size) * 0.5
            
            # Try to use a structured init if possible?
            # For now, random is standard.
            
            try:
                # Use L-BFGS-B
                res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
                
                if res.fun < 1e-8:
                    U = res.x[:size_u*R].reshape(size_u, R)
                    V = res.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = res.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
                
        return None

    # Try to find lower ranks
    # Based on R(3,3,2)=13 and R(3,3,3)=23, R(3,3,5) <= 36.
    # Try 36 first, then 37, 38...
    for R in range(36, 45):
        if time.time() > deadline - 10:
            break
        
        result = solve_rank(R, max_restarts=100)
        if result is not None:
            return result

    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
