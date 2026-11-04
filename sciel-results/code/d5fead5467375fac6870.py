import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 4, 7
    
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    # Precompute target tensor
    # Target[i, j, k, l] = 1 if j == l else 0
    Target = np.zeros((n, m, p, m))
    for j in range(m):
        Target[:, j, :, j] = 1.0
    Target_flat = Target.flatten()

    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        U_3d = U.T.reshape(n, m, r)
        V_3d = V.T.reshape(m, p, r)
        W_3d = W.T.reshape(p, n, r)
        
        # U[i,j,r], V[l,k,r], W[k,i,r] -> result[i,j,k,l]
        computed = np.einsum('ijr,lkr,kir->ijkl', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(computed - Target))
        return max_err < tol

    def residual(params, r):
        U_flat = params[0:r*dim_U]
        V_flat = params[r*dim_U:r*(dim_U+dim_V)]
        W_flat = params[r*(dim_U+dim_V):]
        
        U = U_flat.reshape(r, dim_U).T
        V = V_flat.reshape(r, dim_V).T
        W = W_flat.reshape(r, dim_W).T
        
        U_3d = U.T.reshape(n, m, r)
        V_3d = V.T.reshape(m, p, r)
        W_3d = W.T.reshape(p, n, r)
        
        computed = np.einsum('ijr,lkr,kir->ijkl', U_3d, V_3d, W_3d)
        return (computed - Target).flatten()

    def try_rank(r, n_restarts=5, max_iter=1000):
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            np.random.seed(12345 + restart * 100 + r)
            scale = 0.5
            U_init = np.random.randn(dim_U, r) * scale
            V_init = np.random.randn(dim_V, r) * scale
            W_init = np.random.randn(dim_W, r) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            from scipy.optimize import least_squares
            try:
                sol = least_squares(residual, params_init, args=(r,), method='lm', max_nfev=max_iter)
                if sol.success and np.max(np.abs(sol.fun)) < 1e-5:
                    U_sol = sol.x[0:r*dim_U].reshape(r, dim_U).T
                    V_sol = sol.x[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
                    W_sol = sol.x[r*(dim_U+dim_V):].reshape(r, dim_W).T
                    
                    if verify(U_sol, V_sol, W_sol, tol=1e-4):
                        return (U_sol, V_sol, W_sol)
            except Exception:
                continue
        return None

    # Try ranks from 19 upward
    for r in range(19, 35):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, n_restarts=5, max_iter=1000)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: Schoolbook decomposition (Rank 84)
    rank_school = n * m * p
    U = np.zeros((dim_U, rank_school))
    V = np.zeros((dim_V, rank_school))
    W = np.zeros((dim_W, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
