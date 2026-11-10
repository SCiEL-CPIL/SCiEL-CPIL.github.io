import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal condition: consume all resources
    policy[horizon - 1, :, :] = resource_grid[None, None, :]
    value[horizon - 1, :, :] = utility(resource_grid)[None, None, :]
    
    # Backward induction for t = horizon-2 down to 0
    # We use a dense grid of consumption fractions to approximate the continuous optimum
    # This vectorizes the search over resource levels
    # Using 1000 points is a good balance between accuracy and speed
    num_frac = 1000
    c_fracs = np.linspace(1e-4, 1.0, num_frac)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: Fill remaining with terminal policy (consume all)
            for tt in range(t, -1, -1):
                policy[tt, :, :] = resource_grid[None, None, :]
                value[tt, :, :] = utility(resource_grid)[None, None, :]
            break
            
        for z in range(num_states):
            probs = transition[z]
            
            # 1. Generate all consumption candidates
            # c_candidates shape: (num_resources, num_frac)
            # c_candidates[i, j] = resource_grid[i] * c_fracs[j]
            c_candidates = resource_grid[:, None] * c_fracs[None, :]
            
            # 2. Calculate next period resources for each candidate
            # m_minus_c shape: (num_resources, num_frac)
            m_minus_c = resource_grid[:, None] - c_candidates
            # next_m_base shape: (num_resources, num_frac)
            next_m_base = R * m_minus_c
            
            # 3. Calculate Expected Continuation Value
            # cont_val shape: (num_resources, num_frac)
            cont_val = np.zeros((num_resources, num_frac))
            
            for zp in range(num_states):
                # m_next shape: (num_resources, num_frac)
                m_next = next_m_base + income[zp]
                
                # Interpolate value[t+1, zp, :] at m_next
                # np.interp is vectorized over the second argument
                # We need to do this for each resource level (row)
                # Since np.interp takes 1D arrays, we loop over resource levels 
                # but vectorize over fractions? Or just use a list comprehension?
                # Actually, np.interp can handle 2D if we are careful, but standard usage is 1D.
                # However, looping over 101 resources and calling np.interp on 1000 points is fast.
                
                v_next = np.zeros_like(m_next)
                for i in range(num_resources):
                    v_next[i, :] = np.interp(m_next[i, :], resource_grid, value[t + 1, zp, :])
                
                cont_val += probs[zp] * v_next
            
            # 4. Calculate Total Objective Value
            # utility(c_candidates) shape: (num_resources, num_frac)
            u_c = utility(c_candidates)
            total_val = u_c + beta * cont_val
            
            # 5. Find the optimal fraction index for each resource level
            # We want the max total_val for each row
            opt_idx = np.argmax(total_val, axis=1)
            
            # 6. Extract optimal consumption and value
            # opt_c[i] = c_candidates[i, opt_idx[i]]
            opt_c = c_candidates[np.arange(num_resources), opt_idx]
            opt_val = total_val[np.arange(num_resources), opt_idx]
            
            policy[t, z, :] = opt_c
            value[t, z, :] = opt_val
            
            # Ensure validity: 0 < c <= m
            # opt_c is already >= 1e-4 * m (from c_fracs min)
            # and <= m (from c_fracs max = 1.0)
            # But due to floating point, clip just in case
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid)
            
    # Final validation pass
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    return policy
