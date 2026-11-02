import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (nm, mp, pn)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_exact(U, V, W, tol=1e-9):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=tol): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol): return False
        
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        # Check tensor decomposition
        T_recon = np.einsum('uc,vc,wc->uvw', U_snap, V_snap, W_snap, optimize=True)
        
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    def try_rank(r, time_limit=100):
        start_time = time.time()
        np.random.seed(42)
        
        # Try a few restarts
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(nm, r))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(mp, r))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(pn, r))
            
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    U_flat = params[:nm*r].reshape(nm, r)
                    V_flat = params[nm*r:nm*r + mp*r].reshape(mp, r)
                    W_flat = params[nm*r + mp*r:].reshape(pn, r)
                    
                    T_recon = np.einsum('uc,vc,wc->uvw', U_flat, V_flat, W_flat, optimize=True)
                    error = np.sum((T_recon - T) ** 2)
                    
                    # Penalty for non-half-integers
                    def penalty(M):
                        return np.sum((M * 2 - np.round(M * 2)) ** 2)
                    
                    return error + 0.1 * (penalty(U_flat) + penalty(V_flat) + penalty(W_flat))
                
                def gradient(params):
                    U_flat = params[:nm*r].reshape(nm, r)
                    V_flat = params[nm*r:nm*r + mp*r].reshape(mp, r)
                    W_flat = params[nm*r + mp*r:].reshape(pn, r)
                    
                    T_recon = np.einsum('uc,vc,wc->uvw', U_flat, V_flat, W_flat, optimize=True)
                    R = T_recon - T
                    
                    grad_U = np.einsum('uvw,vc,wc->uc', R, V_flat, W_flat, optimize=True)
                    grad_V = np.einsum('uvw,uc,wc->vc', R, U_flat, W_flat, optimize=True)
                    grad_W = np.einsum('uvw,uc,vc->wc', R, U_flat, V_flat, optimize=True)
                    
                    # Gradient of penalty
                    def grad_penalty(M):
                        return 0.2 * (M * 2 - np.round(M * 2)) # Derivative of (2x - round(2x))^2 is 4(2x-round(2x))*2? No.
                        # Let f(x) = (2x - round(2x))^2. f'(x) = 2(2x - round(2x)) * 2 = 4(2x - round(2x)).
                        # Wait, round is constant locally. So derivative is 4(2x - round(2x)).
                        # In objective, coeff is 0.1. So 0.4(2x - round(2x)).
                        return 0.4 * (M * 2 - np.round(M * 2))

                    grad_U += grad_penalty(U_flat)
                    grad_V += grad_penalty(V_flat)
                    grad_W += grad_penalty(W_flat)

                    return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                res = minimize(objective, params0, jac=gradient, method='BFGS', 
                               options={'maxiter': 20, 'gtol': 1e-4})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:nm*r].reshape(nm, r)
                    V_opt = res.x[nm*r:nm*r + mp*r].reshape(mp, r)
                    W_opt = res.x[nm*r + mp*r:].reshape(pn, r)
                    
                    if check_exact(U_opt, V_opt, W_opt):
                        return U_opt, V_opt, W_opt
                        
            except Exception:
                pass

        return None

    # Deadline guard
    start_total = time.time()
    time_limit_search = 120  # Leave 30s margin
    
    # Try rank 23
    sol_23 = try_rank(23, time_limit_search)
    
    if sol_23 is not None:
        U, V, W = sol_23
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    sol = get_schoolbook()
    return {"U": sol["U"].tolist(), "V": sol["V"].tolist(), "W": sol["W"].tolist()}
