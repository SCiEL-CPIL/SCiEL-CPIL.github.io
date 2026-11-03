import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook
    rank = nm * mp // m # n*m*p? No, n*m*p = 3*3*5 = 45.
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    def residual(x, R):
        # x is flattened U, V, W
        u_size = nm * R
        v_size = mp * R
        w_size = pn * R
        
        U_r = x[:u_size].reshape(nm, R)
        V_r = x[u_size:u_size+v_size].reshape(mp, R)
        W_r = x[u_size+v_size:].reshape(pn, R)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                    res[i, j, k] = val - 1
        return res.flatten()
        
    def try_reduce_rank(R):
        # Initialize with schoolbook, zero out last column
        U_init = np.zeros((nm, R))
        V_init = np.zeros((mp, R))
        W_init = np.zeros((pn, R))
        
        # Copy first R columns from schoolbook
        U_init[:, :R] = U[:, :R]
        V_init[:, :R] = V[:, :R]
        W_init[:, :R] = W[:, :R]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few random perturbations
        for attempt in range(10):
            if attempt > 0:
                noise = np.random.normal(0, 0.1, x0.shape)
                x0_pert = x0 + noise
            else:
                x0_pert = x0
            
            try:
                sol = least_squares(residual, x0_pert, args=(R,), method='trf', max_nfev=1000)
                if sol.success and np.max(np.abs(sol.fun)) < 1e-9:
                    # Solution found
                    u_size = nm * R
                    v_size = mp * R
                    w_size = pn * R
                    U_sol = sol.x[:u_size].reshape(nm, R)
                    V_sol = sol.x[u_size:u_size+v_size].reshape(mp, R)
                    W_sol = sol.x[u_size+v_size:].reshape(pn, R)
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify
                    res_check = residual(np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()]), R)
                    if np.max(np.abs(res_check)) < 1e-9:
                        return U_round, V_round, W_round, R
            except Exception:
                pass
        return None

    # Try to reduce rank
    start_time = time.time()
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try 44
    if time.time() - start_time < 110:
        res = try_reduce_rank(rank - 1)
        if res:
            best_U, best_V, best_W, best_rank = res
            
    # If time permits, try lower
    # ...
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
