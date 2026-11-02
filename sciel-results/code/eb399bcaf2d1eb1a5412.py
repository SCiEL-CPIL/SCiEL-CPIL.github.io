import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 2, 5
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Fallback rank 18 solution
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    def check_exact(U, V, W, rank, tol=1e-5):
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_rank_search(target_rank, max_restarts=20):
        if time.time() > deadline - 10:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 13 + target_rank)
            x0 = np.random.uniform(-1, 1, size=(n*m + m*p + p*n) * target_rank)
            
            def objective(params):
                U = params[:n*m*target_rank].reshape((n*m, target_rank))
                V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
                W = params[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                target = 1.0 if j == jp else 0.0
                                val = 0.0
                                for c in range(target_rank):
                                    val += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                                err += (val - target) ** 2
                return err

            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            
            if res.fun < 1e-8:
                U = res.x[:n*m*target_rank].reshape((n*m, target_rank))
                V = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
                W = res.x[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
                
                U_r = snap_to_half_integers(U)
                V_r = snap_to_half_integers(V)
                W_r = snap_to_half_integers(W)
                
                if U_r is not None and V_r is not None and W_r is not None:
                    if check_exact(U_r, V_r, W_r, target_rank, tol=1e-6):
                        return (U_r, V_r, W_r)
                
                if check_exact(U, V, W, target_rank, tol=1e-6):
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if check_exact(U_i, V_i, W_i, target_rank, tol=1e-6):
                        return (U_i, V_i, W_i)
                    
                    U_r2 = np.round(U * 2) / 2
                    V_r2 = np.round(V * 2) / 2
                    W_r2 = np.round(W * 2) / 2
                    if check_exact(U_r2, V_r2, W_r2, target_rank, tol=1e-6):
                        return (U_r2, V_r2, W_r2)
        
        return None

    # Try rank 17
    if time.time() < deadline - 20:
        res = try_rank_search(17)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 16
    if time.time() < deadline - 20:
        res = try_rank_search(16)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return get_rank_18_solution()
