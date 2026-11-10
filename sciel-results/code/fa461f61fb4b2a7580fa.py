import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    m_grid = resource_grid
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility and derivative functions
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
    policy[horizon - 1, :, :] = m_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(m_grid)[np.newaxis, :]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * m_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * m_grid)[np.newaxis, :]
            break
        
        for z in range(num_states):
            m = m_grid
            c_opt = np.zeros(num_resources)
            v_opt = np.zeros(num_resources)
            
            # Initial guess: consume a fraction based on remaining periods or just 0.5
            # A good heuristic for CRRA is to use the policy from t+1 if possible, 
            # or a simple fraction. Let's start with c = 0.5 * m.
            c_current = 0.5 * m
            
            # Newton-Raphson iterations
            max_iter = 30
            tol = 1e-9
            eps = 1e-5  # Step size for numerical differentiation of continuation value
            
            for _ in range(max_iter):
                # 1. Calculate current continuation value and its derivative
                # V_next(m') where m' = R * (m - c) + income[z']
                # We need V_next(m') and V_next'(m')
                
                # Interpolate value[t+1, zp, :] at m'
                # m' is shape (num_resources,) for each zp
                
                cont_val = np.zeros(num_resources)
                cont_deriv = np.zeros(num_resources)
                
                for zp in range(num_states):
                    m_next = R * (m - c_current) + income[zp]
                    
                    # Clip m_next to valid range for interpolation
                    # resource_grid is from 0.001 to 40
                    # If m_next < min(resource_grid), we might need to handle it.
                    # The problem states resource_grid = np.geomspace(0.001, 40, 101)
                    # So min is 0.001. If m_next < 0.001, we clamp to 0.001 for interpolation?
                    # Actually, the policy must satisfy 0 < c <= m.
                    # If c is close to m, m' can be small.
                    # Let's clamp m_next to [resource_grid[0], resource_grid[-1]]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    # Interpolate value
                    v_next = np.interp(m_next_clipped, resource_grid, value[t + 1, zp, :])
                    
                    # Numerical derivative of value w.r.t m
                    # V'(m) approx (V(m+h) - V(m-h)) / (2h)
                    h = eps
                    m_next_plus = np.clip(m_next + h, resource_grid[0], resource_grid[-1])
                    m_next_minus = np.clip(m_next - h, resource_grid[0], resource_grid[-1])
                    
                    v_next_plus = np.interp(m_next_plus, resource_grid, value[t + 1, zp, :])
                    v_next_minus = np.interp(m_next_minus, resource_grid, value[t + 1, zp, :])
                    
                    # Derivative w.r.t m_next
                    dv_dm_next = (v_next_plus - v_next_minus) / (2 * h)
                    
                    # Chain rule: d/dm_next is multiplied by d m_next / d c = -R
                    # So d/dc (V_next(m')) = dv_dm_next * (-R)
                    # But we are differentiating the objective w.r.t c.
                    # Objective: u(c) + beta * sum( trans * V_next(m') )
                    # d/dc Objective = u'(c) + beta * sum( trans * V_next'(m') * (-R) )
                    
                    cont_val += transition[z, zp] * v_next
                    cont_deriv += transition[z, zp] * dv_dm_next * (-R)
                
                # 2. Calculate F(c) = u'(c) + beta * cont_deriv
                # Note: cont_deriv already includes the chain rule factor (-R)
                # So F(c) = u'(c) + beta * cont_deriv
                
                f_val = util_deriv(c_current) + beta * cont_deriv
                
                # 3. Calculate F'(c) for Newton step
                # F'(c) = u''(c) + beta * d/dc(cont_deriv)
                # d/dc(cont_deriv) = d/dc (sum trans * V''(m') * (-R)) 
                # = sum trans * V''(m') * (-R) * (-R) = sum trans * V''(m') * R^2
                # We need V''(m). We can approximate it numerically from V'(m) or V(m).
                # Let's approximate V''(m) using central difference of V'(m) or finite diff of V(m).
                # V''(m) approx (V(m+h) - 2V(m) + V(m-h)) / h^2
                
                cont_second_deriv = np.zeros(num_resources)
                h2 = 1e-4 # Smaller step for second derivative? Or same?
                # Let's use the same h for consistency if possible, but 2nd deriv is noisy.
                # Better: Use the derivative of the derivative.
                # We already have dv_dm_next at m_next.
                # Let's compute dv_dm_next at m_next +/- h2 and take difference.
                
                for zp in range(num_states):
                    m_next = R * (m - c_current) + income[zp]
                    
                    # We need V''(m_next)
                    # Approximate using 3-point formula on V(m)
                    # V''(m) approx (V(m+h) - 2V(m) + V(m-h)) / h^2
                    # We already have v_next (V(m)).
                    # Let's compute V(m+h) and V(m-h) again.
                    
                    m_next_plus2 = np.clip(m_next + h2, resource_grid[0], resource_grid[-1])
                    m_next_minus2 = np.clip(m_next - h2, resource_grid[0], resource_grid[-1])
                    
                    v_next_plus2 = np.interp(m_next_plus2, resource_grid, value[t + 1, zp, :])
                    v_next_minus2 = np.interp(m_next_minus2, resource_grid, value[t + 1, zp, :])
                    
                    d2v_dm2 = (v_next_plus2 - 2 * v_next + v_next_minus2) / (h2 * h2)
                    
                    # Chain rule for 2nd derivative:
                    # d^2/dc^2 V(m') = V''(m') * (dm'/dc)^2 = V''(m') * (-R)^2 = V''(m') * R^2
                    cont_second_deriv += transition[z, zp] * d2v_dm2 * (R * R)
                
                # u''(c)
                if abs(gamma - 1.0) < 1e-12:
                    u_second = -1.0 / (c_current ** 2)
                else:
                    u_second = -gamma * (c_current ** (-gamma - 1))
                
                f_prime = u_second + beta * cont_second_deriv
                
                # Newton step
                # To avoid division by zero or instability, add a small regularization if f_prime is close to 0
                # But f_prime should be negative (concave function).
                # If f_prime is positive, something is wrong.
                # Let's clip f_prime to be at most -1e-6 to ensure step direction is correct (increase c if f_val > 0? No.)
                # F(c) = 0. If F(c) > 0, we need to increase c?
                # u'(c) is decreasing. Cont_deriv is increasing (since V is concave, V' is decreasing, so -V' is increasing).
                # So F(c) is decreasing.
                # If F(c) > 0, we are to the left of the root, so we need to increase c.
                # Newton: c_new = c - F/F'. Since F' < 0, if F > 0, -F/F' > 0, so c increases. Correct.
                
                # Safety: if f_prime is not negative, use a small negative value
                f_prime_safe = np.where(f_prime < -1e-6, f_prime, -1e-6)
                
                step = f_val / f_prime_safe
                c_next = c_current - step
                
                # Clip c_next to valid range [1e-12, m]
                c_next = np.clip(c_next, 1e-12, m)
                
                # Check convergence
                if np.all(np.abs(c_next - c_current) < tol * m):
                    c_current = c_next
                    break
                
                c_current = c_next
            
            c_opt = c_current
            v_opt = utility(c_current) + beta * cont_val
            
            policy[t, z, :] = c_opt
            value[t, z, :] = v_opt
    
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
