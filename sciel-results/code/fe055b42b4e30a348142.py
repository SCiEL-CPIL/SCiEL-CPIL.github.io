import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Build the target tensor T
    # T[u, v, w] = 1 if consistent, else 0
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
        # params: flattened U, V, W
        # U: num_u * R
        # V: num_v * R
        # W: num_w * R
        R = len(params) // (num_u + num_v + num_w)
        if R == 0:
            return 0.0
        u_size = num_u * R
        v_size = num_v * R
        w_size = num_w * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        # Compute tensor product sum
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
        w_size = num_w * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc - T
        return np.linalg.norm(diff) < 1e-5

    def get_rounded_solution(params, R):
        u_size = num_u * R
        v_size = num_v * R
        w_size = num_w * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        # Round to nearest 0.5
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Check if rounded solution is valid
        T_calc_round = np.einsum('ur, vr, wr -> uvw', U_round, V_round, W_round)
        diff_round = T_calc_round - T
        if np.linalg.norm(diff_round) < 1e-5:
            return U_round, V_round, W_round
        return None

    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Try ranks from low to high, but stop if time is up
    # Schoolbook is 63. We try to find something lower.
    # Based on theoretical bounds, 47 is a good target.
    # We try a range.
    
    best_solution = None
    best_rank = 63
    
    # Schoolbook fallback
    U_school = [[0] * 63 for _ in range(num_u)]
    V_school = [[0] * 63 for _ in range(num_v)]
    W_school = [[0] * 63 for _ in range(num_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Search for lower rank
    # We prioritize finding a valid solution over finding the absolute minimum if time is tight.
    # We try ranks 45 to 62.
    for R in range(45, 63):
        if time.time() > deadline:
            break
            
        found = False
        # Fewer restarts for higher ranks to save time? Or more for lower?
        # Lower rank is harder. Let's do 5 restarts for each rank.
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            U_init = np.random.randn(num_u, R)
            V_init = np.random.randn(num_v, R)
            W_init = np.random.randn(num_w, R)
            
            # Normalize to help optimization
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            # Max iterations limited to ensure we don't hang
            max_iter = 500
            res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})
            
            if res.fun < 1e-4:
                # Check if it's a valid solution
                if check_solution(res.x):
                    rounded = get_rounded_solution(res.x, R)
                    if rounded is not None:
                        U_round, V_round, W_round = rounded
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                    # If rounding fails, we might return the float solution?
                    # The task says "integer/half-integer entries".
                    # So we must round. If rounding fails, this solution is invalid for the contract.
                    # We continue searching.
                    
    # If no better solution found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
