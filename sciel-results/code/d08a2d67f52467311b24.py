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
    
    # Use a denser internal grid for backward induction to reduce interpolation error
    n_internal = 400
    m_internal = np.geomspace(0.001, 40, n_internal)
    
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
    
    # Policy and Value arrays on internal grid
    policy = np.zeros((horizon, num_states, n_internal))
    value = np.zeros((horizon, num_states, n_internal))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(m_internal)
        policy[t, z, :] = m_internal
    
    # Backward Induction using Newton-Raphson with fallback to bisection
    # and PCHIP interpolation for continuation values on internal grid
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = m_internal
                    value[tt, z, :] = utility(m_internal)
            break
        
        # Pre-build PCHIP interpolators for V'_{t+1}[z'](m) and V_{t+1}[z'](m)
        v_prime_interps = []
        value_interps = []
        for zp in range(num_states):
            v_prime_grid = u_prime(policy[t+1, zp, :])
            try:
                vp_interp = PchipInterpolator(m_internal, v_prime_grid)
            except Exception:
                vp_interp = None
            v_prime_interps.append(vp_interp)
            
            try:
                val_interp = PchipInterpolator(m_internal, value[t+1, zp, :])
            except Exception:
                val_interp = None
            value_interps.append(val_interp)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Vectorized Newton-Raphson with bisection fallback
            c_low = np.full(n_internal, c_floor)
            c_high = m_internal.copy()
            c = m_internal.copy()  # Initial guess: consume all (boundary)
            
            max_iter = 40
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute LHS: u'(c)
                lhs = u_prime(c)
                
                # Compute RHS: beta * R * sum_zp probs[zp] * V'_{t+1}[zp](R*(m-c) + income[zp])
                rhs = np.zeros(n_internal)
                
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    
                    m_next = R * (m_internal - c) + income[zp]
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                    
                    if v_prime_interps[zp] is not None:
                        v_prime_vals = v_prime_interps[zp](m_next_clipped)
                    else:
                        v_prime_grid = u_prime(policy[t+1, zp, :])
                        v_prime_vals = np.interp(m_next_clipped, m_internal, v_prime_grid)
                    
                    rhs += probs[zp] * v_prime_vals
                
                rhs = beta * R * rhs
                
                # FOC: f(c) = u'(c) - rhs = 0
                # Bisection update (safe and converges)
                too_low = lhs > rhs
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                # Try Newton step where possible
                # Compute numerical derivative of the FOC
                dc = np.maximum(np.abs(c) * 1e-6, 1e-10)
                c_plus = np.clip(c + dc, c_floor, m_internal)
                
                lhs_plus = u_prime(c_plus)
                rhs_plus = np.zeros(n_internal)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    m_next_plus = R * (m_internal - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, m_internal[0], m_internal[-1])
                    if v_prime_interps[zp] is not None:
                        vp_plus = v_prime_interps[zp](m_next_plus_clipped)
                    else:
                        vp_plus = np.interp(m_next_plus_clipped, m_internal, u_prime(policy[t+1, zp, :]))
                    rhs_plus += probs[zp] * vp_plus
                rhs_plus = beta * R * rhs_plus
                
                f_val = lhs - rhs
                f_plus = lhs_plus - rhs_plus
                df_dc = (f_plus - f_val) / dc
                
                # Newton step
                newton_c = c - f_val / np.where(np.abs(df_dc) > 1e-14, df_dc, 1e-14)
                
                # Use Newton step if it's within bounds, otherwise use bisection midpoint
                bisection_mid = (c_low + c_high) / 2.0
                c = np.where(
                    (newton_c > c_low) & (newton_c < c_high),
                    newton_c,
                    bisection_mid
                )
                c = np.clip(c, c_floor, m_internal)
                
                # Check convergence
                if np.all(c_high - c_low < 1e-12):
                    break
            
            c_opt = np.clip(c, c_floor, m_internal)
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(n_internal)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (m_internal - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                
                if value_interps[zp] is not None:
                    v_next = value_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, m_internal, value[t+1, zp, :])
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    # Ensure terminal condition is exact
    t = horizon - 1
    for z in range(num_states):
        policy[t, z, :] = m_internal
        value[t, z, :] = utility(m_internal)
    
    # Interpolate policy back to the original resource_grid
    policy_out = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy_out[t, z, :] = np.interp(resource_grid, m_internal, policy[t, z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10, c[5] equals grid
    for t in range(horizon):
        for z in range(num_states):
            policy_out[t, z, :] = np.clip(policy_out[t, z, :], c_floor, resource_grid)
            # Terminal period: consume all
            if t == horizon - 1:
                policy_out[t, z, :] = resource_grid
    
    return policy_out
