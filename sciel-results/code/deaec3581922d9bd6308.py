import scipy
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # Indices: A (nm), B (mp), C (pn)
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    # Solver function
    def solve_rank(R, max_restarts=10, timeout=100):
        start_time = time.time()
        best_solution = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout - 10:
                break
                
            # Initialize
            # Use random initialization with small variance
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Gauge fixing: set first non-zero element of each column of W to 1?
            # Or just normalize W columns to unit norm.
            norms = np.linalg.norm(W, axis=0)
            norms[norms == 0] = 1
            W = W / norms
            
            # Reshape for fast computation
            # M_{a,b,r} = U_{a,r} V_{b,r}
            # X_{a,b,c} = sum_r M_{a,b,r} W_{c,r}
            
            def compute_X(U, V, W):
                # U: (nm, R), V: (mp, R), W: (pn, R)
                # M_r = U[:, r] @ V[:, r].T -> (nm, mp)
                # Stack M_r -> (nm, mp, R)
                M = np.einsum('ar,br->ab r', U, V)
                # X_{a,b,c} = sum_r M_{a,b,r} W_{c,r}
                # Reshape M to (nm*mp, R)
                M_flat = M.reshape(-1, R)
                # W is (pn, R)
                # Y_{ab, c} = M_flat @ W.T
                Y = M_flat @ W.T
                # Reshape Y to (nm, mp, pn)
                X = Y.reshape(n*m, m*p, p*n)
                return X

            def error_func(params):
                # params: flattened U, V, W
                U_f, V_f, W_f = params[:n*m*R], params[n*m*R:n*m*R+m*p*R], params[n*m*R+m*p*R:]
                U = U_f.reshape(n*m, R)
                V = V_f.reshape(m*p, R)
                W = W_f.reshape(p*n, R)
                
                X = compute_X(U, V, W)
                err = X - T
                return err.flatten()

            def jacobian_func(params):
                U_f, V_f, W_f = params[:n*m*R], params[n*m*R:n*m*R+m*p*R], params[n*m*R+m*p*R:]
                U = U_f.reshape(n*m, R)
                V = V_f.reshape(m*p, R)
                W = W_f.reshape(p*n, R)
                
                X = compute_X(U, V, W)
                E = X - T # (nm, mp, pn)
                
                # Grad U: dE/dU_{a,r} = 2 * sum_{b,c} E_{a,b,c} V_{b,r} W_{c,r}
                # E_{a,:,:} is (mp, pn)
                # W is (pn, R)
                # Q_{b,r} = sum_c E_{a,b,c} W_{c,r} -> (mp, R)
                # Q = E_{a} @ W
                Q = np.einsum('abc,cr->abr', E, W) # (nm, mp, R)
                # Grad U_{a,r} = sum_b Q_{a,b,r} V_{b,r}
                Grad_U = np.einsum('abr,br->ar', Q, V)
                
                # Grad V: dE/dV_{b,r} = 2 * sum_{a,c} E_{a,b,c} U_{a,r} W_{c,r}
                # P_{a,r} = sum_c E_{a,b,c} W_{c,r}
                P = np.einsum('abc,cr->abr', E, W) # (nm, mp, R) -- wait, index b is fixed?
                # No, for fixed b, we need sum over a, c.
                # Let's swap axes.
                # E_{a,b,c}. Fix b.
                # S_{a,r} = sum_c E_{a,b,c} W_{c,r}
                S = np.einsum('abc,cr->abr', E, W) # (nm, mp, R)
                # Grad V_{b,r} = sum_a S_{a,b,r} U_{a,r}
                Grad_V = np.einsum('abr,ar->br', S, U)
                
                # Grad W: dE/dW_{c,r} = 2 * sum_{a,b} E_{a,b,c} U_{a,r} V_{b,r}
                # T_{a,r} = sum_b E_{a,b,c} V_{b,r}
                T_val = np.einsum('abc,br->acr', E, V) # (nm, pn, R)
                # Grad W_{c,r} = sum_a T_{a,c,r} U_{a,r}
                Grad_W = np.einsum('acr,ar->cr', T_val, U)
                
                # Combine
                grad = np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])
                return grad

            # Use least_squares
            try:
                res = scipy.optimize.least_squares(
                    error_func, 
                    np.concatenate([U.flatten(), V.flatten(), W.flatten()]),
                    jac=jacobian_func,
                    method='trf',
                    max_nfev=5000
                )
                err = np.linalg.norm(res.fun)
                if err < best_error:
                    best_error = err
                    # Extract solution
                    U_sol, V_sol, W_sol = res.x[:n*m*R], res.x[n*m*R:n*m*R+m*p*R], res.x[n*m*R+m*p*R:]
                    best_solution = (U_sol.reshape(n*m, R), V_sol.reshape(m*p, R), W_sol.reshape(p*n, R))
            except Exception as e:
                pass
        
        if best_solution is not None and best_error < 1e-6:
            U_sol, V_sol, W_sol = best_solution
            # Snap to half-integers
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            # Verify
            X_snap = compute_X(U_snap, V_snap, W_snap)
            if np.allclose(X_snap, T, atol=1e-9):
                return U_snap, V_snap, W_snap
        
        return None

    # Try ranks 19 to 23
    for R in range(19, 24):
        sol = solve_rank(R, max_restarts=5, timeout=110)
        if sol is not None:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return {"U": U_school, "V": V_school, "W": W_school}
