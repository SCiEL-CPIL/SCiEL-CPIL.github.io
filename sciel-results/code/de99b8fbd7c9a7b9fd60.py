import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Helper to check if a decomposition is valid
    def check_validity_residual(U, V, W, tol=1e-6):
        R = U.shape[1]
        err = 0.0
        V_mat = V.reshape(m, p, R) # V_mat[j, k, r]
        U_mat = U.reshape(n, m, R) # U_mat[i, j, r]
        W_mat = W.reshape(p, n, R) # W_mat[k, i, r]
        
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        # ... (existing code) ...
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W

    def search_lower_rank(start_rank, end_rank, max_iters=50):
        best_U, best_V, best_W = None, None, None
        best_rank = start_rank
        for R in range(start_rank, end_rank - 1, -1):
            if time.time() > deadline - 10:
                break
            # Try to find a solution of rank R
            # Use random initialization and simple gradient descent
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
                
                # Simple optimization loop
                # We want to minimize sum_{i,k} || U_i diag(W_ki) V_k^T - I ||_F^2
                # This is non-convex.
                # We can use scipy.optimize.minimize on the flattened parameters.
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def objective(params):
                    if time.time() > deadline - 10:
                        return float('inf')
                    U_p = params[:n*m*R].reshape(n, m, R)
                    V_p = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                    W_p = params[n*m*R + m*p*R:].reshape(p, n, R)
                    
                    err = 0.0
                    for i in range(n):
                        for k in range(p):
                            U_i = U_p[i]
                            V_k = V_p[:, k]
                            W_ki = W_p[k, i]
                            M_ik = U_i @ np.diag(W_ki) @ V_k.T
                            err += np.sum((M_ik - np.eye(m))**2)
                    return err

                try:
                    res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': max_iters, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < 1e-4:
                        U_sol = res.x[:n*m*R].reshape(n, m, R).reshape(n*m, R)
                        V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m, p, R).reshape(m*p, R)
                        W_sol = res.x[n*m*R + m*p*R:].reshape(p, n, R).reshape(p*n, R)
                        # Verify
                        err = check_validity_residual(U_sol, V_sol, W_sol)
                        if err < 1e-6:
                            best_U, best_V, best_W = U_sol, V_sol, W_sol
                            best_rank = R
                            break # Found a solution for this rank, try lower
                except Exception:
                    pass
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        return None, None, None, start_rank

    # First, get the known rank 55 solution
    U_55, V_55, W_55 = get_rank_55_solution()
    err_55 = check_validity_residual(U_55, V_55, W_55)
    
    if err_55 > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank
    U_best, V_best, W_best, rank_best = search_lower_rank(54, 40, max_iters=20)
    
    if U_best is not None:
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    else:
        return {"U": U_55.tolist(), "V": V_55.tolist(), "W": W_55.tolist()}
