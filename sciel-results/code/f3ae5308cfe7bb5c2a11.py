import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank search
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Define the cost function
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Flatten to vector for optimizer
    def cost_func(params):
        r = int(np.sqrt(len(params) / (n*m + m*p + p*n))) # This is tricky if r changes
        # Better to pass r as arg or use closure
        pass

    # I will write a helper that takes r and returns a cost function
    def make_cost_func(r):
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        total_size = size_u + size_v + size_w
        
        def cost(params):
            if time.time() > deadline:
                return np.inf
            u_flat = params[:size_u]
            v_flat = params[size_u:size_u+size_v]
            w_flat = params[size_u+size_v:]
            
            U = u_flat.reshape((n*m, r))
            V = v_flat.reshape((m*p, r))
            W = w_flat.reshape((p*n, r))
            
            cost_val = 0.0
            # Sum over i, k, j1, j2
            for i in range(n):
                for k in range(p):
                    w_idx = k * n + i
                    for j1 in range(m):
                        u_idx1 = i * m + j1
                        for j2 in range(m):
                            v_idx = j2 * p + k
                            s = 0.0
                            for c in range(r):
                                s += U[u_idx1, c] * V[v_idx, c] * W[w_idx, c]
                            target = 1.0 if j1 == j2 else 0.0
                            cost_val += (s - target) ** 2
            return cost_val
        return cost

    # Try ranks from 19 to 24
    best_result = None
    best_rank = 24
    best_cost = np.inf
    
    # Pre-define schoolbook as fallback
    U_school = np.zeros((n*m, 24))
    V_school = np.zeros((m*p, 24))
    W_school = np.zeros((p*n, 24))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    fallback = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    for r in range(19, 25):
        if time.time() > deadline:
            break
            
        cost_func = make_cost_func(r)
        
        # Random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            # Initialize with small random values
            np.random.seed(42 + restart)
            size_u = n * m * r
            size_v = m * p * r
            size_w = p * n * r
            x0 = np.random.uniform(-1, 1, size_u + size_v + size_w)
            
            # Use L-BFGS-B
            try:
                res = opt.minimize(cost_func, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                if res.fun < 1e-6:
                    # Found a solution
                    u_flat = res.x[:size_u]
                    v_flat = res.x[size_u:size_u+size_v]
                    w_flat = res.x[size_u+size_v:]
                    
                    U = u_flat.reshape((n*m, r))
                    V = v_flat.reshape((m*p, r))
                    W = w_flat.reshape((p*n, r))
                    
                    # Round to nearest half-integer?
                    # The problem allows integer/half-integer.
                    # If the solution is close to half-integers, round them.
                    # Check if rounding preserves validity.
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    valid = True
                    for i in range(n):
                        for k in range(p):
                            w_idx = k * n + i
                            for j1 in range(m):
                                u_idx1 = i * m + j1
                                for j2 in range(m):
                                    v_idx = j2 * p + k
                                    s = 0.0
                                    for c in range(r):
                                        s += U_round[u_idx1, c] * V_round[v_idx, c] * W_round[w_idx, c]
                                    target = 1.0 if j1 == j2 else 0.0
                                    if abs(s - target) > 1e-6:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    
                    if valid:
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    else:
                        # If rounding fails, try to return the float solution? 
                        # No, must be integer/half-integer.
                        # So we might need to search for integer solutions specifically.
                        pass
                        
            except Exception as e:
                pass

    return fallback
