import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    # Schoolbook rank 24 as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank 22 using least squares with multiple restarts
    from scipy.optimize import least_squares
    
    def try_find_rank22(seed):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Random initialization
        U0 = np.random.randn(6, 22) * 0.5
        V0 = np.random.randn(12, 22) * 0.5
        W0 = np.random.randn(8, 22) * 0.5
        
        def residuals(params):
            U = params[:6*22].reshape(6, 22)
            V = params[6*22:6*22+12*22].reshape(12, 22)
            W = params[6*22+12*22:].reshape(8, 22)
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            return (R - T).flatten()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=500)
            if np.linalg.norm(result.fun) < 1e-8:
                U = result.x[:6*22].reshape(6, 22)
                V = result.x[6*22:6*22+12*22].reshape(12, 22)
                W = result.x[6*22+12*22:].reshape(8, 22)
                
                R = np.einsum('ic,jc,kc->ijk', U, V, W)
                if np.allclose(R, T, atol=1e-8):
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    R_rounded = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(R_rounded, T, atol=1e-8):
                        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    return {"U": U, "V": V, "W": W}
        except:
            pass
        
        return None
    
    # Try multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        result = try_find_rank22(seed)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
