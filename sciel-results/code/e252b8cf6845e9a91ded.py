import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin
    
    def residual(params):
        R = len(params) // (n*m + m*p + p*n)
        if R == 0:
            return np.array([0])
        
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Target is all 1s
        return S.flatten() - 1.0

    def _try_rank(R, n_restarts=5):
        num_vars = R * (n*m + m*p + p*n)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(restart * 100 + R)
            
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Gauge fixing
            U[0, :] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residual, params, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if np.linalg.norm(sol.fun) < 1e-8:
                U_sol = sol.x[0:n*m*R].reshape(n*m, R)
                V_sol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            s = np.dot(U_snap[u_idx, :] * V_snap[v_idx, :], W_snap[w_idx, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return (U_snap, V_snap, W_snap)
        
        return None

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, column] = 1
                    V[j*p + k, column] = 1
                    W[k*n + i, column] = 1
                    column += 1
        return (U, V, W)
    
    # Try ranks from 45 up to 53
    for R in range(45, 54):
        if time.time() > deadline:
            break
        
        result = _try_rank(R, n_restarts=5)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
