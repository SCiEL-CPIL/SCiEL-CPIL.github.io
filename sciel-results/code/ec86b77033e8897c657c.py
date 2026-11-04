"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triples = len(triples)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    best_rank = n * m * p
    best_solution = None
    
    # Schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    for col, (i, j, k) in enumerate(triples):
        U_sb[u_idx(i,j), col] = 1.0
        V_sb[v_idx(j,k), col] = 1.0
        W_sb[w_idx(k,i), col] = 1.0
    best_solution = (U_sb, V_sb, W_sb)
    
    def compute_residual(U, V, W):
        res = np.zeros(n_triples)
        for t, (i, j, k) in enumerate(triples):
            res[t] = np.dot(U[u_idx(i,j), :], V[v_idx(j,k), :] * W[w_idx(k,i), :]) - 1.0
        return res
    
    def try_rank_lm(rank, n_restarts=50):
        if time.time() > deadline - 15:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 7919 + rank * 104729)
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def res_func(params):
                U_p = params[:n*m*rank].reshape(n*m, rank)
                V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                return compute_residual(U_p, V_p, W_p)
            
            try:
                result = least_squares(res_func, params, method='lm', max_nfev=300)
            except:
                continue
            
            max_err = np.max(np.abs(result.fun))
            if max_err < 1e-8:
                U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                return (U_sol, V_sol, W_sol)
        
        return None
    
    def try_rank_alternating(rank, n_restarts=30):
        if time.time() > deadline - 15:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 31337 + rank * 271828)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            for it in range(150):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[v_idx(j,k), :] * W[w_idx(k,i), :]
                        try:
                            u, *_ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_new[u_idx(i,j), :] = u
                        except:
                            U_new[u_idx(i,j), :] = U[u_idx(i,j), :]
                U = U_new
                
                # Solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[u_idx(i,j), :] * W[w_idx(k,i), :]
                        try:
                            v, *_ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_new[v_idx(j,k), :] = v
                        except:
                            V_new[v_idx(j,k), :] = V[v_idx(j,k), :]
                V = V_new
                
                # Solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[u_idx(i,j), :] * V[v_idx(j,k), :]
                        try:
                            w, *_ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_new[w_idx(k,i), :] = w
                        except:
                            W_new[w_idx(k,i), :] = W[w_idx(k,i), :]
                W = W_new
                
                # Check
                res = compute_residual(U, V, W)
                if np.max(np.abs(res)) < 1e-8:
                    return (U, V, W)
        
        return None
    
    lower_bound = max(n*m, m*p, n*p)
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 25:
            break
        
        # Try LM first
        sol = try_rank_lm(rank, n_restarts=40)
        if sol is None:
            # Try alternating
            sol = try_rank_alternating(rank, n_restarts=20)
        
        if sol is not None:
            best_rank = rank
            best_solution = sol
            break
    
    U, V, W = best_solution
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
