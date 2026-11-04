import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    def get_rank_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            # M1
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M2
            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1
            # M3
            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M4
            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            # M5
            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            # M6
            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M7
            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank, max_attempts=5, max_nfev=500):
        if time.time() > deadline - 15:
            return None
        
        U_size = n * m
        V_size = m * p
        W_size = p * n
        total_vars = (U_size + V_size + W_size) * rank
        
        # Precompute index triples for faster residual computation
        idx_triples = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
        
        def residuals(params):
            U = params[:U_size*rank].reshape(U_size, rank)
            V = params[U_size*rank:(U_size+V_size)*rank].reshape(V_size, rank)
            W = params[(U_size+V_size)*rank:].reshape(W_size, rank)
            
            res = np.empty(len(idx_triples))
            for idx, (ui, vi, wi) in enumerate(idx_triples):
                res[idx] = np.dot(U[ui], V[vi] * W[wi]) - 1.0
            return res

        best_solution = None
        best_cost = np.inf
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            x0 = np.random.uniform(-0.5, 0.5, size=total_vars)
            
            try:
                sol = least_squares(residuals, x0, method='trf', max_nfev=max_nfev)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_solution = sol.x
                    
                    if best_cost < 1e-10:
                        U = best_solution[:U_size*rank].reshape(U_size, rank)
                        V = best_solution[U_size*rank:(U_size+V_size)*rank].reshape(V_size, rank)
                        W = best_solution[(U_size+V_size)*rank:].reshape(W_size, rank)
                        
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                            
            except Exception:
                continue
                
        return None

    # Try lower ranks with increasing effort
    for rank in [36, 34, 32, 30]:
        result = try_rank(rank)
        if result is not None:
            return result
            
    return get_rank_42()
