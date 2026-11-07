import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    consumption = np.zeros((horizon, Z, M))
    V = np.zeros((horizon, Z, M))
    
    def utility(c):
        c_safe = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    V[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, np.newaxis, :]
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Use a fine grid for consumption fraction s = c/m
    # 2000 points is a good balance between accuracy and speed
    S = 2000
    s_grid = np.linspace(1e-4, 1.0, S)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with c=m (valid but suboptimal)
            for tt in range(t, -1, -1):
                consumption[tt, :, :] = resource_grid[np.newaxis, np.newaxis, :]
            return consumption

        V_next = V[t + 1]
        
        for z in range(Z):
            m_col = resource_grid[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]
            
            c_candidates = m_col * s_row
            c_candidates = np.maximum(c_candidates, 1e-12)
            
            obj_values = np.zeros((M, S))
            
            for zp in range(Z):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                
                m_prime = R * (resource_grid[:, np.newaxis] - c_candidates) + income[zp]
                
                # Clip m_prime to valid range for interpolation
                # np.interp handles out-of-bounds by using endpoint values, which is correct here
                m_prime_flat = m_prime.flatten()
                v_interp = np.interp(m_prime_flat, resource_grid, V_next[zp, :])
                v_interp = v_interp.reshape(M, S)
                
                obj_values += prob * v_interp
            
            u_c = utility(c_candidates)
            total_obj = u_c + beta * obj_values
            
            best_idx = np.argmax(total_obj, axis=1)
            best_s = s_grid[best_idx]
            best_c = resource_grid * best_s
            
            # Vectorized local refinement using ternary search on a small window
            # This is much faster than scipy.minimize_scalar for 2000*6*2 evaluations
            
            # Window: 20 points around the best guess
            # Step size in s_grid
            if S > 1:
                s_step = s_grid[1] - s_grid[0]
            else:
                s_step = 1.0
            
            # Define search window in terms of s indices
            # We search in c-space, but it's linear in s for fixed m
            # Let's just do a small grid search around best_c for each m_i
            
            # To keep it fast, we do a vectorized small grid refinement
            # Refine around best_c with a smaller step
            # Number of refinement points
            N_REF = 21
            # Range: +/- 2 original steps
            ref_range = 4 * s_step * resource_grid[:, np.newaxis]
            ref_s = best_s[:, np.newaxis] + np.linspace(-2*s_step, 2*s_step, N_REF)[np.newaxis, :]
            ref_s = np.clip(ref_s, 1e-4, 1.0)
            
            ref_c = resource_grid[:, np.newaxis] * ref_s
            ref_c = np.maximum(ref_c, 1e-12)
            
            ref_obj = np.zeros((M, N_REF))
            
            for zp in range(Z):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                
                m_prime = R * (resource_grid[:, np.newaxis] - ref_c) + income[zp]
                m_prime_flat = m_prime.flatten()
                v_interp = np.interp(m_prime_flat, resource_grid, V_next[zp, :])
                v_interp = v_interp.reshape(M, N_REF)
                
                ref_obj += prob * v_interp
            
            ref_u = utility(ref_c)
            ref_total = ref_u + beta * ref_obj
            
            ref_best_idx = np.argmax(ref_total, axis=1)
            
            # Extract best c and V
            c_opt = np.array([ref_c[i, ref_best_idx[i]] for i in range(M)])
            v_opt = np.array([ref_total[i, ref_best_idx[i]] for i in range(M)])
            
            # Final clip to ensure validity
            c_opt = np.clip(c_opt, 1e-12, resource_grid)
            
            consumption[t, z, :] = c_opt
            V[t, z, :] = v_opt

    return consumption
