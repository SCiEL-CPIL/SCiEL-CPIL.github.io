import numpy as np
import scipy.optimize
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook Rank 12
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    deadline = time.time() + 110 # Safety margin

    # Try ranks 9 and 10
    for r in [9, 10]:
        if time.time() > deadline:
            break
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            if r == 9:
                break # Found minimal known rank

    return best_result

def _find_decomposition(n, m, p, r, deadline):
    np.random.seed(42)
    
    def loss_and_jac(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r : n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r :]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        residuals = np.zeros(n*m*p)
        idx = 0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j, :]
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    
                    val = np.dot(u_row, v_row * w_row)
                    err = val - 1.0
                    residuals[idx] = err
                    idx += 1
                    
                    grad_U[i*m+j, :] += 2 * err * (v_row * w_row)
                    grad_V[j*p+k, :] += 2 * err * (u_row * w_row)
                    grad_W[k*n+i, :] += 2 * err * (u_row * v_row)
        
        loss = np.sum(residuals**2)
        jac = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, jac

    for attempt in range(100):
        if time.time() > deadline:
            return None
            
        # Random initialization
        U_init = np.random.randn(n * m, r)
        V_init = np.random.randn(m * p, r)
        W_init = np.random.randn(p * n, r)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.minimize(
                loss_and_jac, x0, method='L-BFGS-B', jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.fun < 1e-10:
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try rounding to integers or half-integers
                for scale in [1, 2]:
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    
                    # Verify
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                                err += (val - 1.0)**2
                                
                    if err < 1e-8:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        except Exception:
            continue
            
    return None
