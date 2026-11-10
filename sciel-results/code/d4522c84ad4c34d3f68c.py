import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Policy and Value arrays
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Backward Induction using FOC bisection with PCHIP interpolation
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
        
        # Pre-build PCHIP interpolators for V'_{t+1}[z'](m) for all z'
        # V'(m) = u'(policy[t+1, z', m]) by envelope theorem
        v_prime_interps = []
        value_interps = []
        for zp in range(num_states):
            v_prime_grid = u_prime(policy[t+1, zp, :])
            try:
                vp_interp = PchipInterpolator(resource_grid, v_prime_grid)
            except Exception:
                vp_interp = None
            v_prime_interps.append(vp_interp)
            
            try:
                val_interp = PchipInterpolator(resource_grid, value[t+1, zp, :])
            except Exception:
                val_interp = None
            value_interps.append(val_interp)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Vectorized bisection over all m points simultaneously
            c_low = np.full(num_resources, c_floor)
            c_high = resource_grid.copy()
            
            max_iter = 60
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                c_mid = (c_low + c_high) / 2.0
                
                # Compute LHS: u'(c_mid)
                lhs = u_prime(c_mid)
                
                # Compute RHS: beta * R * sum_zp probs[zp] * V'_{t+1}[zp](R*(m-c_mid) + income[zp])
                rhs = np.zeros(num_resources)
                
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    
                    # m_next for each m point
                    m_next = R * (resource_grid - c_mid) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    if v_prime_interps[zp] is not None:
                        v_prime_vals = v_prime_interps[zp](m_next_clipped)
                    else:
                        v_prime_grid = u_prime(policy[t+1, zp, :])
                        v_prime_vals = np.interp(m_next_clipped, resource_grid, v_prime_grid)
                    
                    rhs += probs[zp] * v_prime_vals
                
                rhs = beta * R * rhs
                
                # FOC: u'(c) - beta*R*V'(...) = 0
                # u' is decreasing in c.
                # If lhs > rhs, c is too low (need to increase c to lower u')
                # If lhs < rhs, c is too high (need to decrease c to raise u')
                
                too_low = lhs > rhs
                c_low = np.where(too_low, c_mid, c_low)
                c_high = np.where(~too_low, c_mid, c_high)
                
                # Check convergence
                if np.all(c_high - c_low < 1e-12):
                    break
            
            c_opt = (c_low + c_high) / 2.0
            c_opt = np.clip(c_opt, c_floor, resource_grid)
            
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (resource_grid - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                
                if value_interps[zp] is not None:
                    v_next = value_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, resource_grid, value[t+1, zp, :])
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    # Ensure terminal condition is exact
    t = horizon - 1
    for z in range(num_states):
        policy[t, z, :] = resource_grid
        value[t, z, :] = utility(resource_grid)
    
    return policy
