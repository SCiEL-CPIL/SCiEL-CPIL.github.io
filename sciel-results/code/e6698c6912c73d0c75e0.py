import numpy as np
import time
import sys
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 6]

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
    
    start_time = time.time()
    time_limit = 145  # 5s margin before 150s kill

    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Tensor T_sub of shape (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=float)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0

    def verify_decomposition(U, V, W, T_target, atol=1e-9):
        T_comp = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.allclose(T_comp, T_target, atol=atol)

    def try_find_decomposition(r, max_time=30):
        if time.time() - start_time > time_limit:
            return None
            
        # Values in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
        vals = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        
        def objective(params):
            U_flat = params[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
            V_flat = params[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
            W_flat = params[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
            
            T_comp = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
            diff = T_comp - T_sub
            return np.sum(diff * diff)
        
        def snap_to_half_int(x):
            return np.round(x * 2) / 2
        
        best_res = None
        best_val = float('inf')
        
        # Strategy 1: Try structured initializations
        # For 2x3 x 3x3, schoolbook has rank 18
        # Try to find rank 9 or lower
        
        # Approach: Use the fact that we can decompose into rank-1 terms
        # Each rank-1 term picks (i,j), (j,k), (k,i)
        
        # Try small ranks first with more careful search
        num_restarts = 100
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Different initialization strategies
            if restart < 30:
                # Sparse random: mostly zeros, some ±1
                U = np.zeros((n_sub * m_sub, r))
                V = np.zeros((m_sub * p_sub, r))
                W = np.zeros((p_sub * n_sub, r))
                for t in range(r):
                    # Pick a random (i,j) for U
                    idx_u = np.random.randint(0, n_sub * m_sub)
                    U[idx_u, t] = np.random.choice([-1, 1])
                    # Pick a random (j,k) for V consistent with U's j
                    j_val = idx_u // m_sub
                    idx_v_candidates = [j_val * p_sub + k for k in range(p_sub)]
                    idx_v = np.random.choice(idx_v_candidates)
                    V[idx_v, t] = np.random.choice([-1, 1])
                    # Pick a random (k,i) for W consistent
                    k_val = idx_v % p_sub
                    i_val = idx_u // m_sub
                    idx_w = k_val * n_sub + i_val
                    W[idx_w, t] = np.random.choice([-1, 1])
            elif restart < 60:
                # Random half-integers but sparser
                U = np.zeros((n_sub * m_sub, r))
                V = np.zeros((m_sub * p_sub, r))
                W = np.zeros((p_sub * n_sub, r))
                for t in range(r):
                    if np.random.random() < 0.7:
                        idx_u = np.random.randint(0, n_sub * m_sub)
                        U[idx_u, t] = np.random.choice(vals)
                    if np.random.random() < 0.7:
                        idx_v = np.random.randint(0, m_sub * p_sub)
                        V[idx_v, t] = np.random.choice(vals)
                    if np.random.random() < 0.7:
                        idx_w = np.random.randint(0, p_sub * n_sub)
                        W[idx_w, t] = np.random.choice(vals)
            else:
                # Fully random half-integers
                U = np.random.choice(vals, size=(n_sub * m_sub, r))
                V = np.random.choice(vals, size=(m_sub * p_sub, r))
                W = np.random.choice(vals, size=(p_sub * n_sub, r))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_val:
                    best_val = res.fun
                    best_res = res
                    
                    # Check if we found exact solution
                    if best_val < 1e-6:
                        U_opt = best_res.x[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
                        V_opt = best_res.x[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
                        W_opt = best_res.x[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
                        
                        U_int = snap_to_half_int(U_opt)
                        V_int = snap_to_half_int(V_opt)
                        W_int = snap_to_half_int(W_opt)
                        
                        if verify_decomposition(U_int, V_int, W_int, T_sub):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            except Exception:
                pass
        
        # Final check on best result
        if best_res is not None and best_val < 1e-4:
            U_opt = best_res.x[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
            V_opt = best_res.x[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
            W_opt = best_res.x[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
            
            U_int = snap_to_half_int(U_opt)
            V_int = snap_to_half_int(V_opt)
            W_int = snap_to_half_int(W_opt)
            
            if verify_decomposition(U_int, V_int, W_int, T_sub):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
        return None

    # Try ranks from 9 upwards (schoolbook for 2x3x3 is 18)
    for r in [9, 10, 11, 12, 13, 14, 15, 16]:
        if time.time() - start_time > time_limit - 5:
            break
        res = try_find_decomposition(r)
        if res:
            U_sub, V_sub, W_sub = res
            global_rank = 2 * r
            U_global = [[0] * global_rank for _ in range(n * m)]
            V_global = [[0] * global_rank for _ in range(m * p)]
            W_global = [[0] * global_rank for _ in range(p * n)]
            
            # First block: columns 0..r-1
            # Second block: columns r..2r-1
            for t in range(r):
                for i in range(n * m):
                    U_global[i][t] = U_sub[i][t]
                    U_global[i][t + r] = U_sub[i][t]
                    
            for t in range(r):
                for j in range(m_sub * p_sub):
                    V_global[j][t] = V_sub[j][t]
                    V_global[j + m_sub * p_sub][t + r] = V_sub[j][t]
                    
            for t in range(r):
                for k in range(p_sub * n_sub):
                    W_global[k][t] = W_sub[k][t]
                    W_global[k + p_sub * n_sub][t + r] = W_sub[k][t]
                    
            # Verify global decomposition
            U_g = np.array(U_global, dtype=float)
            V_g = np.array(V_global, dtype=float)
            W_g = np.array(W_global, dtype=float)
            
            # Full tensor T of shape (n*m, m*p, p*n)
            T_full = np.zeros((n * m, m * p, p * n), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_full[i * m + j, j * p + k, k * n + i] = 1.0
            
            T_check = np.einsum('it,jt,kt->ijk', U_g, V_g, W_g)
            if np.allclose(T_check, T_full, atol=1e-9):
                return {"U": U_global, "V": V_global, "W": W_global}
            
            # If verification failed, continue to next rank
            continue
            
    return {"U": U_school, "V": V_school, "W": W_school}
