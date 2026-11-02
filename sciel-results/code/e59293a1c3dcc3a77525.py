import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Build the target tensor T
    # T[u, v, w] = 1 if consistent, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
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
    target_norm = np.linalg.norm(T_flat)
    
    def objective(params):
        # params: flattened U, V, W
        # U: num_u * R
        # V: num_v * R
        # W: num_w * R
        R = len(params) // (num_u + num_v + num_w)
        if R == 0:
            return 0.0
        u_size = num_u * R
        v_size = num_v * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        # Compute tensor product sum
        # T_calc[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        T_calc_flat = T_calc.flatten()
        
        diff = T_calc_flat - T_flat
        return np.dot(diff, diff)

    def check_solution(params):
        R = len(params) // (num_u + num_v + num_w)
        if R == 0:
            return False
        u_size = num_u * R
        v_size = num_v * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc - T
        return np.linalg.norm(diff) < 1e-5

    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Try ranks from a lower bound up to 62
    # A trivial lower bound is max(n*m, m*p, p*n) ? No.
    # For 3x3x3, rank is 7 (Strassen).
    # For 3x3x7, let's try from 20 upwards.
    # We prioritize finding *any* solution below 63.
    
    best_result = None
    
    for R in range(20, 63):
        if time.time() > deadline - 10: # Leave 10s for fallback
            break
            
        # Try a few restarts for each rank
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R * 100 + restart)
            U_init = np.random.randn(num_u, R)
            V_init = np.random.randn(num_v, R)
            W_init = np.random.randn(num_w, R)
            
            # Scale to avoid large numbers initially
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B for speed
            res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.fun < 1e-6:
                if check_solution(res.x):
                    # Try to round to half-integers
                    U_opt = res.x[:num_u*R].reshape(num_u, R)
                    V_opt = res.x[num_u*R:num_u*R+num_v*R].reshape(num_v, R)
                    W_opt = res.x[num_u*R+num_v*R:].reshape(num_w, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Check if rounded solution is valid
                    T_calc_round = np.einsum('ur, vr, wr -> uvw', U_round, V_round, W_round)
                    diff_round = T_calc_round - T
                    if np.linalg.norm(diff_round) < 1e-5:
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                    else:
                        # If rounding fails, we might have a real solution but not an exact integer one.
                        # The task requires exact integer/half-integer.
                        # We cannot return a non-integer solution.
                        # We continue searching.
                        pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
