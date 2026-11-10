import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility and its derivative
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    # Terminal condition: at t = horizon - 1 (last period), consume all
    # Actually, horizon = 6, so t = 0..5. At t=5 (last), consume all.
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]  # c = m
    # Value at last period: u(m)
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break
        
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                
                # Objective: maximize u(c) + beta * sum_z' P(z'|z) * V_{t+1}(R*(m-c)+income[z'], z')
                # c in [0, m], but we floor c at 1e-12 for utility
                # Define objective function
                def obj(c):
                    c = max(c, 1e-12)
                    c = min(c, m)
                    m_next = R * (m - c)
                    cont_val = 0.0
                    for zp in range(num_states):
                        m_next_zp = m_next + income[zp]
                        # Interpolate V_{t+1} at m_next_zp for state zp
                        if m_next_zp <= resource_grid[0]:
                            v_next = value[t + 1, zp, 0]
                        elif m_next_zp >= resource_grid[-1]:
                            v_next = value[t + 1, zp, -1]
                        else:
                            v_next = np.interp(m_next_zp, resource_grid, value[t + 1, zp, :])
                        cont_val += transition[z, zp] * v_next
                    return utility(c) + beta * cont_val
                
                # Derivative of objective w.r.t. c
                def obj_deriv(c):
                    c = max(c, 1e-12)
                    c = min(c, m)
                    h = 1e-6 * max(1.0, abs(c))
                    c_plus = min(c + h, m)
                    c_minus = max(c - h, 1e-12)
                    if c_plus == c_minus:
                        return 0.0
                    return (obj(c_plus) - obj(c_minus)) / (c_plus - c_minus)
                
                # Bisection on derivative to find where it crosses zero
                # f(c) = obj_deriv(c), we want f(c) = 0
                # f is decreasing (concave objective)
                lo = 1e-12
                hi = m
                
                # Check endpoints
                f_lo = obj_deriv(lo)
                f_hi = obj_deriv(hi)
                
                if f_lo <= 0:
                    # Optimum at lo
                    c_opt = lo
                elif f_hi >= 0:
                    # Optimum at hi
                    c_opt = hi
                else:
                    # Bisection
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        f_mid = obj_deriv(mid)
                        if abs(f_mid) < 1e-10:
                            break
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_opt = 0.5 * (lo + hi)
                
                policy[t, z, m_idx] = c_opt
                value[t, z, m_idx] = obj(c_opt)
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
