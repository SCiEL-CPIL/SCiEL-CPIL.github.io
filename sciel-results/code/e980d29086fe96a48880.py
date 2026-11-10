import numpy as np
import time
from scipy.interpolate import PchipInterpolator

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
    n_work = 501
    work_grid = np.geomspace(0.001, 40, n_work)
    
    # Utility and marginal utility
    if gamma == 1:
        def utility(c):
            c = np.maximum(c, c_floor)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, c_floor)
            return 1.0 / c
        def u_double_prime(c):
            c = np.maximum(c, c_floor)
            return -1.0 / (c * c)
    else:
        def utility(c):
            c = np.maximum(c, c_floor)
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            c = np.maximum(c, c_floor)
            return np.power(c, -gamma)
        def u_double_prime(c):
            c = np.maximum(c, c_floor)
            return -gamma * np.power(c, -gamma - 1)
    
    policy_work = np.zeros((horizon, num_states, n_work))
    value_work = np.zeros((horizon, num_states, n_work))
    
    # Terminal condition: consume all
    t = horizon - 1
    for z in range(num_states):
        value_work[t, z, :] = utility(work_grid)
        policy_work[t, z, :] = work_grid
    
    # Build PCHIP interpolators for terminal period
    v_interp_next = [PchipInterpolator(work_grid, value_work[t, z, :]) for z in range(num_states)]
    
    # Backward induction via solving the Euler equation
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        # Precompute V'_{t+1} using PCHIP derivative
        vp_next = np.empty((num_states, n_work))
        for zp in range(num_states):
            vp_next[zp, :] = v_interp_next[zp](work_grid, 1)
        
        for z in range(num_states):
            probs = transition[z]
            nonzero_zp = [zp for zp in range(num_states) if probs[zp] > 0]
            
            # Initial guess: use a heuristic saving rule
            m_ratio = work_grid / (work_grid + 1.0)
            saving_rate = 0.25 * (1.0 - m_ratio) + 0.05
            c = np.clip(work_grid * (1.0 - saving_rate), c_floor, work_grid)
            
            c_low = np.full(n_work, c_floor)
            c_high = work_grid.copy()
            
            max_iter = 80
            tol = 1e-13
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute RHS of FOC: beta * R * E[V'(m_next)]
                rhs = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next = R * (work_grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                    vprime_vals = vp_next[zp, :]
                    # Use PCHIP for smooth interpolation of V'
                    # But vp_next is already on work_grid, so use np.interp for speed
                    # Actually, let's use the interpolator for better accuracy
                    vprime_vals = v_interp_next[zp](m_next_clipped, 1)
                    rhs += probs[zp] * vprime_vals
                
                rhs = beta * R * rhs
                
                # LHS = u'(c)
                lhs = u_prime(c)
                
                # f(c) = lhs - rhs
                f_val = lhs - rhs
                
                # Bisection update
                too_low = f_val > 0  # c too low, need higher c
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                # Check convergence
                if np.all(c_high - c_low < tol):
                    break
                
                # Newton step with numerical derivative
                dc = np.maximum(np.abs(c) * 1e-6, 1e-10)
                c_plus = np.clip(c + dc, c_floor, work_grid)
                
                rhs_plus = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next_plus = R * (work_grid - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, work_grid[0], work_grid[-1])
                    vprime_vals_plus = v_interp_next[zp](m_next_plus_clipped, 1)
                    rhs_plus += probs[zp] * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus
                
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
                v_next = v_interp_next[zp](m_next_clipped, 0)
                cont_val += probs[zp] * v_next
            value_work[t, z, :] = u_val + beta * cont_val
        
        # Build new PCHIP interpolators for next iteration
        v_interp_next = [PchipInterpolator(work_grid, value_work[t, z, :]) for z in range(num_states)]
    
    # Interpolate from work grid to original grid
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            if t == horizon - 1:
                policy[t, z, :] = orig_grid
            else:
                # Use PCHIP for smoother interpolation
                interp = PchipInterpolator(work_grid, policy_work[t, z, :])
                policy[t, z, :] = interp(orig_grid)
                policy[t, z, :] = np.clip(policy[t, z, :], c_floor, orig_grid)
    
    # Final validity checks
    for z in range(num_states):
        policy[horizon-1, z, :] = orig_grid
        policy[horizon-1, z, :] = np.clip(policy[horizon-1, z, :], c_floor, orig_grid)
    
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.maximum(policy[t, z, :], c_floor)
            policy[t, z, :] = np.minimum(policy[t, z, :], orig_grid + 1e-10)
    
    return policy
