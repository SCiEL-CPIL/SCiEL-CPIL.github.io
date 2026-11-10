import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    num_states = len(income)
    orig_grid = resource_grid
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Use a very dense internal grid for accurate interpolation
    n_work = 2001
    work_grid = np.geomspace(0.001, 40, n_work)
    
    # Utility and marginal utility functions
    if gamma == 1:
        def utility(c):
            c = np.maximum(c, c_floor)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, c_floor)
            return 1.0 / c
    else:
        def utility(c):
            c = np.maximum(c, c_floor)
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            c = np.maximum(c, c_floor)
            return np.power(c, -gamma)
    
    policy_work = np.zeros((horizon, num_states, n_work))
    value_work = np.zeros((horizon, num_states, n_work))
    
    # Terminal condition: consume all
    t = horizon - 1
    for z in range(num_states):
        value_work[t, z, :] = utility(work_grid)
        policy_work[t, z, :] = work_grid
    
    # Backward induction: solve Euler equation at each grid point
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        for z in range(num_states):
            probs = transition[z]
            nonzero_zp = [zp for zp in range(num_states) if probs[zp] > 1e-15]
            
            # Initial guess: use a smooth saving rule
            # For CRRA, optimal consumption is roughly c = m * (1 - s)
            # where s is a saving rate that depends on m
            m = work_grid
            # Heuristic: saving rate is higher for lower m
            # Use a logistic-type function
            saving_rate = 0.5 / (1.0 + 0.5 * m)  # ranges from ~0.5 to ~0
            c = np.clip(m * (1.0 - saving_rate), c_floor, m)
            
            # Bisection bounds
            c_low = np.full(n_work, c_floor)
            c_high = m.copy()
            
            max_iter = 80
            tol = 1e-12
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute RHS of FOC: beta * R * E[V'(m_next)]
                rhs = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next = R * (m - c) + income[zp]
                    # Clip to work grid range
                    m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                    # Interpolate V' from next period
                    vprime_vals = np.interp(m_next_clipped, work_grid, 
                                           u_prime(policy_work[t+1, zp, :]))
                    rhs += probs[zp] * vprime_vals
                rhs = beta * R * rhs
                
                # LHS = u'(c)
                lhs = u_prime(c)
                
                # f(c) = lhs - rhs
                # f is decreasing in c: as c increases, lhs decreases, 
                # m_next decreases, V'(m_next) increases, rhs increases
                # So f > 0 means c too low, f < 0 means c too high
                
                too_low = lhs > rhs
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                # Check convergence
                if np.all(c_high - c_low < tol):
                    break
                
                # Newton step with numerical derivative
                dc = np.maximum(np.abs(c) * 1e-6, 1e-12)
                c_plus = np.clip(c + dc, c_floor, m)
                
                rhs_plus = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next_plus = R * (m - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, work_grid[0], work_grid[-1])
                    vprime_vals_plus = np.interp(m_next_plus_clipped, work_grid, 
                                               u_prime(policy_work[t+1, zp, :]))
                    rhs_plus += probs[zp] * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus
                
                f_val = lhs - rhs
                f_plus = u_prime(c_plus) - rhs_plus
                df_dc = (f_plus - f_val) / dc
                
                # Newton step
                safe_df = np.where(np.abs(df_dc) > 1e-16, df_dc, 1e-16)
                newton_c = c - f_val / safe_df
                
                # Use Newton if within bounds, else bisection
                bisection_mid = (c_low + c_high) / 2.0
                c = np.where(
                    (newton_c > c_low) & (newton_c < c_high),
                    newton_c,
                    bisection_mid
                )
                c = np.clip(c, c_floor, m)
            
            policy_work[t, z, :] = c
            
            # Compute value
            u_val = utility(c)
            cont_val = np.zeros(n_work)
            for zp in nonzero_zp:
                m_next = R * (m - c) + income[zp]
                m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                v_next = np.interp(m_next_clipped, work_grid, value_work[t+1, zp, :])
                cont_val += probs[zp] * v_next
            value_work[t, z, :] = u_val + beta * cont_val
    
    # Interpolate from work grid to original grid
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
            # Enforce bounds: 0 < c <= m
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, orig_grid)
    
    # Final validity checks
    # Terminal period must equal grid
    for z in range(num_states):
        policy[horizon-1, z, :] = orig_grid
        policy[horizon-1, z, :] = np.clip(policy[horizon-1, z, :], c_floor, orig_grid)
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.maximum(policy[t, z, :], c_floor)
            policy[t, z, :] = np.minimum(policy[t, z, :], orig_grid + 1e-10)
    
    return policy
