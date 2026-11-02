import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    def search_decomp(n_sub, m_sub, p_sub, min_rank, max_rank, seed_offset=0, time_limit=30.0):
        num_u_sub = n_sub * m_sub
        num_v_sub = m_sub * p_sub
        num_w_sub = p_sub * n_sub
        
        T_sub = np.zeros((num_u_sub, num_v_sub, num_w_sub))
        for u in range(num_u_sub):
            i = u // m_sub
            j = u % m_sub
            for v in range(num_v_sub):
                j_v = v // p_sub
                k = v % p_sub
                if j != j_v:
                    continue
                for w in range(num_w_sub):
                    k_w = w // n_sub
                    i_w = w % n_sub
                    if k == k_w and i == i_w:
                        T_sub[u, v, w] = 1.0
                        
        T_flat = T_sub.flatten()
        
        def objective(params):
            R = len(params) // (num_u_sub + num_v_sub + num_w_sub)
            if R == 0:
                return 0.0
            u_size = num_u_sub * R
            v_size = num_v_sub * R
            w_size = num_w_sub * R
            
            U = params[:u_size].reshape(num_u_sub, R)
            V = params[u_size:u_size+v_size].reshape(num_v_sub, R)
            W = params[u_size+v_size:].reshape(num_w_sub, R)
            
            T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
            diff = T_calc.flatten() - T_flat
            return np.dot(diff, diff)
            
        def check_solution(U, V, W):
            T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
            return np.linalg.norm(T_calc - T_sub) < 1e-6
            
        def is_half_integer(arr):
            doubled = arr * 2
            return np.allclose(doubled, np.round(doubled), atol=1e-6)
            
        local_start = time.time()
        local_deadline = local_start + time_limit
        
        for R in range(min_rank, max_rank + 1):
            if time.time() > local_deadline or time.time() > deadline:
                break
            
            found_valid = False
            max_restarts = 20
            
            for restart in range(max_restarts):
                if time.time() > local_deadline or time.time() > deadline:
                    break
                
                np.random.seed(42 + R * 1000 + restart + seed_offset)
                U_init = np.random.randn(num_u_sub, R)
                V_init = np.random.randn(num_v_sub, R)
                W_init = np.random.randn(num_w_sub, R)
                
                U_init /= np.linalg.norm(U_init) + 1e-10
                V_init /= np.linalg.norm(V_init) + 1e-10
                W_init /= np.linalg.norm(W_init) + 1e-10
                
                params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    res = minimize(objective, params_init, method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                    
                    if res.fun < 1e-8:
                        U_opt = res.x[:num_u_sub*R].reshape(num_u_sub, R)
                        V_opt = res.x[num_u_sub*R:num_u_sub*R+num_v_sub*R].reshape(num_v_sub, R)
                        W_opt = res.x[num_u_sub*R+num_u_sub*R+num_v_sub*R:].reshape(num_w_sub, R)
                        
                        if check_solution(U_opt, V_opt, W_opt):
                            U_round = np.round(U_opt * 2) / 2
                            V_round = np.round(V_opt * 2) / 2
                            W_round = np.round(W_opt * 2) / 2
                            
                            if check_solution(U_round, V_round, W_round) and \
                               is_half_integer(U_round) and is_half_integer(V_round) and is_half_integer(W_round):
                                found_valid = True
                                return U_round, V_round, W_round
                except Exception:
                    pass
            
            if found_valid:
                break
        
        return None

    # Search for 3x3x3
    U_333, V_333, W_333 = search_decomp(3, 3, 3, 23, 27, seed_offset=0, time_limit=40.0)
    
    # Search for 3x3x4
    U_334, V_334, W_334 = search_decomp(3, 3, 4, 28, 36, seed_offset=100, time_limit=40.0)
    
    if U_333 is not None and U_334 is not None:
        R3 = U_333.shape[1]
        R4 = U_334.shape[1]
        TotalRank = R3 + R4
        
        U_full = np.zeros((9, TotalRank))
        V_full = np.zeros((21, TotalRank))
        W_full = np.zeros((21, TotalRank))
        
        U_full[:, :R3] = U_333
        for j in range(3):
            for k in range(3):
                V_full[j*7 + k, :R3] = V_333[j*3 + k, :R3]
        
        for k in range(3):
            for i in range(3):
                W_full[k*3 + i, :R3] = W_333[k*3 + i, :R3]
        
        U_full[:, R3:] = U_334
        for j in range(3):
            for k_prime in range(4):
                k = k_prime + 3
                V_full[j*7 + k, R3:] = V_334[j*4 + k_prime, :]
        
        for k_prime in range(4):
            k = k_prime + 3
            for i in range(3):
                W_full[k*3 + i, R3:] = W_334[k_prime*3 + i, :]
        
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        rank_schoolbook = n * m * p
        U_sb = np.zeros((num_u, rank_schoolbook))
        V_sb = np.zeros((num_v, rank_schoolbook))
        W_sb = np.zeros((num_w, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
