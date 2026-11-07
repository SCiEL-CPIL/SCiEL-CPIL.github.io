"""Improved program: piecewise log-linear value function with Newton Euler solve."""
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Precompute utility and marginal utility functions
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def mu(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Terminal condition: at t=horizon-1 (last period), consume all wealth
    # V_{H-1}(m, z) = u(m), c_{H-1}(m, z) = m
    
    # Backward induction with vectorized operations
    # Store value function on grid for each state
    V_next = np.zeros((n_states, n_grid))
    for z in range(n_states):
        V_next[z, :] = util(resource_grid)
    
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon-1, :, :] = resource_grid
    
    # Precompute log of resource grid for log-linear interpolation
    log_grid = np.log(resource_grid)
    
    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon-2, -1, -1):
        V_curr = np.zeros((n_states, n_grid))
        
        for z in range(n_states):
            # Vectorize over all m values
            m_vals = resource_grid
            
            # Initial guess: use a heuristic based on beta and gamma
            # For CRRA, c* ≈ m * (1 - beta*R)^{-1/gamma} approximately, but with income shocks
            # Use a simpler heuristic: c = m * 0.5 as starting point
            c_vals = np.clip(m_vals * 0.5, 1e-12, m_vals)
            
            # Newton iterations
            for _ in range(15):
                # Compute continuation value and derivative for all m simultaneously
                cont_val = np.zeros(n_grid)
                cont_deriv = np.zeros(n_grid)
                
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    
                    # m_prime = R * (m - c) + income[zp] for all m
                    m_prime = R * (m_vals - c_vals) + income[zp]
                    
                    # Clip m_prime to grid range
                    m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                    
                    # Log-linear interpolation of V_next[zp]
                    log_m_prime = np.log(m_prime_clipped)
                    
                    # Find interpolation indices
                    idx = np.searchsorted(log_grid, log_m_prime)
                    idx = np.clip(idx, 1, n_grid - 1)
                    idx0 = idx - 1
                    idx1 = idx
                    
                    # Interpolation weights in log space
                    w0 = log_grid[idx1] - log_m_prime
                    w1 = log_m_prime - log_grid[idx0]
                    wsum = w0 + w1
                    w0 = w0 / wsum
                    w1 = w1 / wsum
                    
                    # Interpolated values
                    v0 = V_next[zp, idx0]
                    v1 = V_next[zp, idx1]
                    v_interp = w0 * v0 + w1 * v1
                    
                    # Derivative: dV/dm = (dV/dlog_m) * (1/m)
                    # dV/dlog_m ≈ (v1 - v0) / (log_grid[idx1] - log_grid[idx0])
                    dlog_grid = log_grid[idx1] - log_grid[idx0]
                    dV_dlog_m = (v1 - v0) / dlog_grid
                    v_deriv = dV_dlog_m / m_prime_clipped
                    
                    cont_val += p * v_interp
                    cont_deriv += p * v_deriv
                
                # FOC: mu(c) - beta * R * cont_deriv = 0
                f = mu(c_vals) - beta * R * cont_deriv
                
                # Compute df/dc numerically using central differences
                dc = 1e-6 * np.maximum(1.0, c_vals)
                c_plus = np.minimum(c_vals + dc, m_vals)
                c_minus = np.maximum(c_vals - dc, 1e-12)
                
                # Compute f at c_plus
                cont_val_plus = np.zeros(n_grid)
                cont_deriv_plus = np.zeros(n_grid)
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    m_prime_plus = R * (m_vals - c_plus) + income[zp]
                    m_prime_plus_clipped = np.clip(m_prime_plus, resource_grid[0], resource_grid[-1])
                    log_m_prime_plus = np.log(m_prime_plus_clipped)
                    idx_plus = np.searchsorted(log_grid, log_m_prime_plus)
                    idx_plus = np.clip(idx_plus, 1, n_grid - 1)
                    idx0_plus = idx_plus - 1
                    idx1_plus = idx_plus
                    w0_plus = log_grid[idx1_plus] - log_m_prime_plus
                    w1_plus = log_m_prime_plus - log_grid[idx0_plus]
                    wsum_plus = w0_plus + w1_plus
                    w0_plus = w0_plus / wsum_plus
                    w1_plus = w1_plus / wsum_plus
                    v0_plus = V_next[zp, idx0_plus]
                    v1_plus = V_next[zp, idx1_plus]
                    v_interp_plus = w0_plus * v0_plus + w1_plus * v1_plus
                    dlog_grid_plus = log_grid[idx1_plus] - log_grid[idx0_plus]
                    dV_dlog_m_plus = (v1_plus - v0_plus) / dlog_grid_plus
                    v_deriv_plus = dV_dlog_m_plus / m_prime_plus_clipped
                    cont_val_plus += p * v_interp_plus
                    cont_deriv_plus += p * v_deriv_plus
                f_plus = mu(c_plus) - beta * R * cont_deriv_plus
                
                # Compute f at c_minus
                cont_val_minus = np.zeros(n_grid)
                cont_deriv_minus = np.zeros(n_grid)
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    m_prime_minus = R * (m_vals - c_minus) + income[zp]
                    m_prime_minus_clipped = np.clip(m_prime_minus, resource_grid[0], resource_grid[-1])
                    log_m_prime_minus = np.log(m_prime_minus_clipped)
                    idx_minus = np.searchsorted(log_grid, log_m_prime_minus)
                    idx_minus = np.clip(idx_minus, 1, n_grid - 1)
                    idx0_minus = idx_minus - 1
                    idx1_minus = idx_minus
                    w0_minus = log_grid[idx1_minus] - log_m_prime_minus
                    w1_minus = log_m_prime_minus - log_grid[idx0_minus]
                    wsum_minus = w0_minus + w1_minus
                    w0_minus = w0_minus / wsum_minus
                    w1_minus = w1_minus / wsum_minus
                    v0_minus = V_next[zp, idx0_minus]
                    v1_minus = V_next[zp, idx1_minus]
                    v_interp_minus = w0_minus * v0_minus + w1_minus * v1_minus
                    dlog_grid_minus = log_grid[idx1_minus] - log_grid[idx0_minus]
                    dV_dlog_m_minus = (v1_minus - v0_minus) / dlog_grid_minus
                    v_deriv_minus = dV_dlog_m_minus / m_prime_minus_clipped
                    cont_val_minus += p * v_interp_minus
                    cont_deriv_minus += p * v_deriv_minus
                f_minus = mu(c_minus) - beta * R * cont_deriv_minus
                
                df_dc = (f_plus - f_minus) / (c_plus - c_minus)
                
                # Newton step
                mask = np.abs(df_dc) > 1e-15
                c_new = c_vals.copy()
                c_new[mask] = c_vals[mask] - f[mask] / df_dc[mask]
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_vals)) < 1e-10 * np.max(m_vals):
                    c_vals = c_new
                    break
                
                c_vals = c_new
            
            consumption[t, z, :] = c_vals
            V_curr[z, :] = util(c_vals) + beta * cont_val
        
        V_next = V_curr
    
    # Ensure final period constraint
    consumption[horizon-1, :, :] = resource_grid
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            c = consumption[t, z, :]
            m = resource_grid
            consumption[t, z, :] = np.clip(c, 1e-12, m)
    
    return consumption
