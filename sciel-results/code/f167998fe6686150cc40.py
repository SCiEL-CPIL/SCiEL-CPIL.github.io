import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Helper to compute tensor error
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape
        U_resh = U.reshape(n, m, -1)
        V_resh = V.reshape(m, p, -1)
        W_resh = W.reshape(p, n, -1)
        
        # W needs to be W[k, i, r] -> W_resh[k, i, r]
        # We want T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        R = U.shape[1]
        T = np.zeros((n, m, p))
        
        for r in range(R):
            Ur = U_resh[:, :, r] # (n, m)
            Vr = V_resh[:, :, r] # (m, p)
            Wr = W_resh[:, :, r] # (p, n) -> We need W[k, i]
            
            # T[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
            # Ur[:, :, None] * Vr[None, :, :] * Wr.T[:, None, :] ?
            # Wr is (p, n). Wr.T is (n, p).
            # We need index i from Ur, j from Ur/Vr, k from Vr/Wr.
            # Term: Ur[i, j] * Vr[j, k] * Wr.T[i, k]
            
            # Broadcast:
            # Ur: (n, m, 1)
            # Vr: (1, m, p)
            # Wr.T: (n, 1, p)
            T += Ur[:, :, np.newaxis] * Vr[np.newaxis, :, :] * Wr.T[:, np.newaxis, :]
            
        return np.sum((T - 1.0)**2)

    # Helper to snap to half-integers and verify
    def snap_and_verify(U, V, W):
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Check if close to original
        if np.max(np.abs(U - U_snapped)) > 1e-3: return None
        if np.max(np.abs(V - V_snapped)) > 1e-3: return None
        if np.max(np.abs(W - W_snapped)) > 1e-3: return None
        
        # Verify exactness with snapped values
        err = compute_error(U_snapped, V_snapped, W_snapped)
        if err < 1e-10:
            return U_snapped, V_snapped, W_snapped
        return None

    # Baseline Construction (Block Decomposition)
    # ... (same as before, but using lists) ...
    
    # Convert baseline to numpy for error check?
    # Actually, just return baseline if search fails.
    
    # Search
    best_result = None
    
    # Try ranks from 116 down to 90
    for R in range(116, 89, -1):
        if time.time() > deadline - 10:
            break
            
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Flatten for optimizer
            def objective_flat(x):
                U_f = x[:n*m*R].reshape(n*m, R)
                V_f = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_f = x[n*m*R + m*p*R :].reshape(p*n, R)
                return compute_error(U_f, V_f, W_f)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective_flat, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    snapped = snap_and_verify(U_opt, V_opt, W_opt)
                    if snapped:
                        return {
                            "U": snapped[0].tolist(),
                            "V": snapped[1].tolist(),
                            "W": snapped[2].tolist()
                        }
            except Exception:
                pass

    # Return Baseline
    # ... (construct baseline and return) ...
