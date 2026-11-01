import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[a_idx, b_idx, c_idx] = 1 if a_idx, b_idx, c_idx correspond to the same (i,j,k)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1

    def reconstruct(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Res = sum_r U[:, r] outer V[:, r] outer W[:, r]
        # Efficiently: Res[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        R = U.shape[1]
        Res = np.zeros((n * m, m * p, p * n))
        for r in range(R):
            Res += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return Res

    def objective(params, R):
        U = params[:n * m * R].reshape(n * m, R)
        V = params[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
        W = params[n * m * R + m * p * R :].reshape(p * n, R)
        Res = reconstruct(U, V, W)
        return np.sum((Res - T) ** 2)

    def is_valid(U, V, W):
        # Check if entries are integers or half-integers
        for mat in [U, V, W]:
            for val in mat:
                # Check if val is close to k/2
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
        # Check exact reconstruction
        Res = reconstruct(U, V, W)
        if np.max(np.abs(Res - T)) > 1e-6:
            return False
        return True

    # Try to find lower rank
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try ranks from 41 down to 35
    for R in range(rank_school - 1, 34, -1):
        if time.time() > deadline:
            break
            
        # A few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with random values or truncated schoolbook
            # Truncated schoolbook is a better start
            U_init = U_school[:, :R]
            V_init = V_school[:, :R]
            W_init = W_school[:, :R]
            
            # Add small noise
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params_init, args=(R,), method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.success:
                    U_sol = res.x[:n * m * R].reshape(n * m, R)
                    V_sol = res.x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                    W_sol = res.x[n * m * R + m * p * R :].reshape(p * n, R)
                    
                    if is_valid(U_sol, V_sol, W_sol):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                pass

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
