import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Build the target tensor
    T = np.zeros((num_u, num_v, num_w))
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    
    def objective(params):
        R = len(params) // (num_u + num_v + num_w)
        if R == 0:
            return 0.0
        u_size = num_u * R
        v_size = num_v * R
        w_size = num_w * R
        
        U = params[:u_size].reshape(num_u, R)
        V = params[u_size:u_size+v_size].reshape(num_v, R)
        W = params[u_size+v_size:].reshape(num_w, R)
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc.flatten() - T_flat
        return np.dot(diff, diff)
    
    def check_solution(U, V, W):
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        return np.linalg.norm(T_calc - T) < 1e-6
    
    def is_half_integer(arr):
        """Check if all entries are integer or half-integer"""
        doubled = arr * 2
        return np.allclose(doubled, np.round(doubled), atol=1e-6)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # First, create the schoolbook solution as fallback
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
    
    best_solution = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound and go up
    # For 3x3x7, try ranks from 25 to 62
    for R in range(25, min(62, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        found_valid = False
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 1000 + restart)
            U_init = np.random.randn(num_u, R)
            V_init = np.random.randn(num_v, R)
            W_init = np.random.randn(num_w, R)
            
            # Normalize to prevent numerical issues
            U_init /= np.linalg.norm(U_init) + 1e-10
            V_init /= np.linalg.norm(V_init) + 1e-10
            W_init /= np.linalg.norm(W_init) + 1e-10
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params_init, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:num_u*R].reshape(num_u, R)
                    V_opt = res.x[num_u*R:num_u*R+num_v*R].reshape(num_v, R)
                    W_opt = res.x[num_u*R+num_u*R+num_v*R:].reshape(num_w, R)
                    
                    # Check if the solution is valid
                    if check_solution(U_opt, V_opt, W_opt):
                        # Try to round to half-integers
                        U_round = np.round(U_opt * 2) / 2
                        V_round = np.round(V_opt * 2) / 2
                        W_round = np.round(W_opt * 2) / 2
                        
                        if check_solution(U_round, V_round, W_round) and \
                           is_half_integer(U_round) and is_half_integer(V_round) and is_half_integer(W_round):
                            found_valid = True
                            best_rank = R
                            best_solution = {
                                "U": U_round.tolist(),
                                "V": V_round.tolist(),
                                "W": W_round.tolist()
                            }
                            break
                        else:
                            # If rounding fails, try with integers
                            U_int = np.round(U_opt)
                            V_int = np.round(V_opt)
                            W_int = np.round(W_opt)
                            
                            if check_solution(U_int, V_int, W_int):
                                found_valid = True
                                best_rank = R
                                best_solution = {
                                    "U": U_int.tolist(),
                                    "V": V_int.tolist(),
                                    "W": W_int.tolist()
                                }
                                break
            except Exception:
                pass
        
        if found_valid:
            # Found a valid solution at this rank, can stop
            break
    
    return best_solution
