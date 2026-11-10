import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 138  # Safety margin before 150s limit
    
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
    
    # Use a denser working grid for better accuracy
    n_work = 401
    work_grid = np.geomspace(0.001, 40, n_work)
    
    # Utility and marginal utility
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
    
    # Backward induction via solving the Euler equation
    # FOC: u'(c) = beta * R * sum_{z'} pi(z'|z) * V'_{t+1}(z', R*(m-c) + y_{z'})
    # We solve this using a damped fixed-point / bisection hybrid on the working grid.
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        # Precompute V'_{t+1} on work_grid for each z'
        vp_next = np.empty((num_states, n_work))
        for zp in range(num_states):
            vp_next[zp, :] = u_prime(policy_work[t+1, zp, :])
        
        for z in range(num_states):
            probs = transition[z]
            # Precompute non-zero transitions
            nonzero_zp = [zp for zp in range(num_states) if probs[zp] > 0]
            
            # Initial guess: use a simple saving rule
            # c ~ m * (1 - s) where s depends on m
            # For low m, save more; for high m, save less
            c = np.empty(n_work)
            
            # Bisection bounds
            c_low = np.full(n_work, c_floor)
            c_high = work_grid.copy()
            
            # Initial guess: linear saving rule
            # c = m * (1 - theta) where theta is a saving rate
            # Use a heuristic: saving rate decreases with m
            # theta ~ R/(1+R) for small m, smaller for large m
            m_ratio = work_grid / (work_grid + 1.0)
            saving_rate = 0.3 * (1.0 - m_ratio) + 0.05  # between 0.35 and 0.05
            c = np.clip(work_grid * (1.0 - saving_rate), c_floor, work_grid)
            
            max_iter = 60
            tol = 1e-11
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute RHS of FOC
                rhs = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next = R * (work_grid - c) + income[zp]
                    # Clip to work grid range for interpolation
                    m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                    # Interpolate V'
                    vprime_vals = np.interp(m_next_clipped, work_grid, vp_next[zp, :])
                    rhs += probs[zp] * vprime_vals
                
                rhs = beta * R * rhs
                
                # LHS = u'(c)
                lhs = u_prime(c)
                
                # f(c) = lhs - rhs; f is decreasing in c (u' decreasing, and more c means less m_next so lower V')
                # Actually: as c increases, lhs decreases, and m_next decreases, so V' increases, so rhs increases
                # f(c) = u'(c) - beta*R*E[V'(m_next)]. Both u' and the expected V' are decreasing in m_next, but
                # u' is decreasing in c directly, and V' is increasing in c (since m_next decreases).
                # So f'(c) = u''(c) - beta*R*E[V''(m_next)*dm_next/dc]
                # u''(c) < 0, and V''(m_next) < 0, dm_next/dc = -R < 0
                # So V''(m_next)*dm_next/dc > 0, so -beta*R*E[positive] < 0
                # f'(c) = negative + negative < 0. So f is strictly decreasing in c.
                
                # Bisection: f(c) > 0 means c is too low (u'(c) too high relative to rhs)
                # Wait: if c is small, u'(c) is large (high marginal utility), and m_next is large so V'(m_next) is small.
                # So f(c) = large - small > 0. We need larger c.
                # If c is large, u'(c) is small, m_next is small, V'(m_next) is large, so f(c) = small - large < 0.
                # So f > 0 => increase c, f < 0 => decrease c.
                
                too_low = lhs > rhs  # f > 0, need higher c
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                # Check convergence
                if np.all(c_high - c_low < tol):
                    break
                
                # Newton step with numerical derivative
                dc = np.maximum(np.abs(c) * 1e-5, 1e-10)
                c_plus = np.clip(c + dc, c_floor, work_grid)
                
                rhs_plus = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next_plus = R * (work_grid - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, work_grid[0], work_grid[-1])
                    vprime_vals_plus = np.interp(m_next_plus_clipped, work_grid, vp_next[zp, :])
                    rhs_plus += probs[zp] * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus
                
                f_val = lhs - rhs
                f_plus = u_prime(c_plus) - rhs_plus
                df_dc = (f_plus - f_val) / dc
                
                # Newton step
                safe_df = np.where(np.abs(df_dc) > 1e-14, df_dc, 1e-14)
                newton_c = c - f_val / safe_df
                
                # Use Newton if within bounds, else bisection
                bisection_mid = (c_low + c_high) / 2.0
                c = np.where(
                    (newton_c > c_low) & (newton_c < c_high),
                    newton_c,
                    bisection_mid
                )
                c = np.clip(c, c_floor, work_grid)
            
            policy_work[t, z, :] = c
            
            # Compute value
            u_val = utility(c)
            cont_val = np.zeros(n_work)
            for zp in nonzero_zp:
                m_next = R * (work_grid - c) + income[zp]
                m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                v_next = np.interp(m_next_clipped, work_grid, value_work[t+1, zp, :])
                cont_val += probs[zp] * v_next
            value_work[t, z, :] = u_val + beta * cont_val
    
    # Interpolate from work grid to original grid
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            # Interpolate policy
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
            # Ensure terminal condition is exact
            if t == horizon - 1:
                policy[t, z, :] = orig_grid
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
