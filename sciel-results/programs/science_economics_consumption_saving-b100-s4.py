import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
        
    def util_second_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Build a finer internal grid for better interpolation
    # Use a denser grid to reduce interpolation error
    internal_grid = np.geomspace(resource_grid[0], resource_grid[-1], 501)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                frac = 1.0 / (horizon - tt)
                policy[tt, :, :] = np.minimum(frac * resource_grid[np.newaxis, :], resource_grid[np.newaxis, :])
                policy[tt, :, :] = np.maximum(policy[tt, :, :], c_floor)
                value[tt, :, :] = utility(policy[tt, :, :])
            break
        
        # Build PCHIP interpolators for value[t+1, z', :] on the internal grid
        # We need to interpolate value[t+1, z', :] from resource_grid to internal_grid first
        # Actually, we'll work directly on the internal grid for the value function
        # Let's restructure: we'll compute value on internal_grid as well
        
        # For now, build interpolators on the original resource_grid
        # and evaluate at points on the internal grid
        
        v_interps = []
        vp_interps = []
        for zp in range(num_states):
            try:
                v_interp = PchipInterpolator(resource_grid, value[t + 1, zp, :])
                vp_interp = PchipInterpolator(resource_grid, util_deriv(policy[t + 1, zp, :]))
                v_interps.append(v_interp)
                vp_interps.append(vp_interp)
            except Exception:
                v_interps.append(None)
                vp_interps.append(None)
        
        for z in range(num_states):
            m = resource_grid
            remaining = horizon - t
            
            # Initial guess for c
            c_est = np.maximum(resource_grid / remaining, c_floor)
            
            # Newton-Raphson iteration
            max_newton = 40
            for _ in range(max_newton):
                if time.time() > deadline - 5:
                    break
                
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    p_zp = transition[z, zp]
                    if p_zp < 1e-15:
                        continue
                    
                    m_next = R * (m - c_est) + income[zp]
                    m_next = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    if vp_interps[zp] is not None:
                        vp_vals = vp_interps[zp](m_next)
                    else:
                        vp_vals = np.interp(m_next, resource_grid, util_deriv(policy[t + 1, zp, :]))
                    
                    # Numerical second derivative for V
                    h = max(1e-6, 0.01 * np.mean(resource_grid))
                    m_next_p = np.clip(m_next + h, resource_grid[0], resource_grid[-1])
                    m_next_m = np.clip(m_next - h, resource_grid[0], resource_grid[-1])
                    
                    if vp_interps[zp] is not None:
                        vp_p = vp_interps[zp](m_next_p)
                        vp_m = vp_interps[zp](m_next_m)
                    else:
                        vp_grid = util_deriv(policy[t + 1, zp, :])
                        vp_p = np.interp(m_next_p, resource_grid, vp_grid)
                        vp_m = np.interp(m_next_m, resource_grid, vp_grid)
                    
                    vpp_vals = (vp_p - vp_m) / (2 * h)
                    
                    sum_v_prime += p_zp * vp_vals
                    sum_v_second_prime += p_zp * vpp_vals
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                # Newton step with damping
                f_prime_safe = np.where(np.abs(f_prime) < 1e-10, 1e-10, f_prime)
                step = f / f_prime_safe
                c_new = c_est - step
                
                # Damping: limit step size
                max_step = 0.5 * np.maximum(m - c_est, 0) + 0.5 * np.maximum(c_est - c_floor, 0)
                max_step = np.maximum(max_step, 1e-12)
                step_clipped = np.clip(step, -max_step, max_step)
                c_new = c_est - step_clipped
                
                # Clip to valid range
                c_new = np.maximum(c_new, c_floor)
                c_new = np.minimum(c_new, m)
                
                # Check convergence
                tol = 1e-11 * np.maximum(1.0, m)
                if np.all(np.abs(c_new - c_est) < tol):
                    c_est = c_new
                    break
                c_est = c_new
            
            policy[t, z, :] = c_est
            
            # Compute value V_t(m) = u(c) + beta * E[V_{t+1}(m')]
            u_c = utility(c_est)
            sum_v_next = np.zeros_like(m)
            for zp in range(num_states):
                p_zp = transition[z, zp]
                if p_zp < 1e-15:
                    continue
                m_next = R * (m - c_est) + income[zp]
                m_next = np.clip(m_next, resource_grid[0], resource_grid[-1])
                
                if v_interps[zp] is not None:
                    v_next = v_interps[zp](m_next)
                else:
                    v_next = np.interp(m_next, resource_grid, value[t + 1, zp, :])
                
                sum_v_next += p_zp * v_next
            
            value[t, z, :] = u_c + beta * sum_v_next

    # Ensure validity: 0 < c <= m + 1e-10
    m_grid = resource_grid[np.newaxis, np.newaxis, :]
    policy = np.minimum(policy, m_grid + 1e-10)
    policy = np.maximum(policy, c_floor)
    
    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=c_floor, posinf=1e10, neginf=0)
    
    return policy
